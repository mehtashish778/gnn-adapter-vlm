from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ChairAttributeDataset(Dataset):
    def __init__(
        self,
        split_path: Path,
        label_vocab_path: Path,
        repo_root: Path,
        image_cache_dir: Path,
    ) -> None:
        payload = load_json(split_path)
        if "samples" not in payload:
            raise RuntimeError(f"Invalid split file format: {split_path}")
        self.samples: List[Dict[str, Any]] = payload["samples"]
        self.label_vocab: Dict[str, int] = load_json(label_vocab_path)
        self.num_labels = len(self.label_vocab)
        self.repo_root = repo_root
        self.image_cache_dir = image_cache_dir
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.corrupt_image_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, row: Dict[str, Any]) -> Path:
        local_rel = row.get("image_path", "")
        if local_rel:
            local = self.repo_root / local_rel
            if local.exists():
                return local
        image_id = str(row.get("image_id"))
        return self.image_cache_dir / f"{image_id}.jpg"

    def _candidate_urls(self, row: Dict[str, Any]) -> List[str]:
        primary = str(row.get("image_url") or "").strip()
        if not primary:
            return []
        candidates = [primary]
        if "/VG_100K/" in primary:
            candidates.append(primary.replace("/VG_100K/", "/VG_100K_2/"))
        elif "/VG_100K_2/" in primary:
            candidates.append(primary.replace("/VG_100K_2/", "/VG_100K/"))
        deduped: List[str] = []
        seen = set()
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped

    def _download_if_missing(
        self, row: Dict[str, Any], path: Path, force: bool = False
    ) -> None:
        if path.exists() and not force:
            return
        urls = self._candidate_urls(row)
        if not urls:
            raise RuntimeError(f"Missing image_url for image_id={row.get('image_id')}")
        last_error: Exception | None = None
        for url in urls:
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=30) as response:
                    image_bytes = response.read()
                with Image.open(BytesIO(image_bytes)) as probe:
                    probe.verify()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    handle.write(image_bytes)
                return
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                UnidentifiedImageError,
            ) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to download valid image for image_id={row.get('image_id')} from candidate URLs."
        ) from last_error

    def _open_local_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _load_image(self, row: Dict[str, Any]) -> Image.Image:
        image_path = self._resolve_image_path(row)
        self._download_if_missing(row, image_path)
        try:
            return self._open_local_image(image_path)
        except (UnidentifiedImageError, OSError):
            self.corrupt_image_count += 1
            if image_path.exists():
                image_path.unlink(missing_ok=True)
            try:
                self._download_if_missing(row, image_path, force=True)
                return self._open_local_image(image_path)
            except Exception:
                return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _labels_to_multihot(self, labels: List[str]) -> torch.Tensor:
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        for label in labels:
            if label in self.label_vocab:
                target[self.label_vocab[label]] = 1.0
        return target

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        image = self._load_image(row)
        labels = row.get("attributes_norm", [])
        target = self._labels_to_multihot(labels)
        target_mask = torch.ones(self.num_labels, dtype=torch.float32)
        return {
            "image": image,
            "target": target,
            "target_mask": target_mask,
            "image_id": int(row["image_id"]),
            "labels": labels,
        }


def collate_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "images": [item["image"] for item in batch],
        "targets": torch.stack([item["target"] for item in batch], dim=0),
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item.get("image_path", "") for item in batch],
        "labels": [item["labels"] for item in batch],
    }
    out["target_mask"] = torch.stack([item["target_mask"] for item in batch], dim=0)
    return out


class MultiLabelImageDataset(Dataset):
    """Generic multi-label dataset: local images + split JSON."""

    def __init__(
        self,
        split_path: Path,
        label_vocab_path: Path,
        repo_root: Path,
        image_root: Path,
        *,
        allow_url_download: bool = True,
        positives_key: str = "attributes_norm",
    ) -> None:
        payload = load_json(split_path)
        if "samples" not in payload:
            raise RuntimeError(f"Invalid split file format: {split_path}")
        self.samples: List[Dict[str, Any]] = payload["samples"]
        self.label_vocab: Dict[str, int] = load_json(label_vocab_path)
        self.num_labels = len(self.label_vocab)
        self.repo_root = repo_root
        self.image_root = image_root.resolve()
        self.allow_url_download = allow_url_download
        self.positives_key = positives_key
        self.image_cache_dir = self.image_root
        self.corrupt_image_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, row: Dict[str, Any]) -> Path:
        local_rel = row.get("image_path", "")
        if local_rel:
            local = self.image_root / local_rel
            if local.exists():
                return local
        if not self.allow_url_download:
            return self.image_root / str(local_rel).lstrip("/")
        image_id = str(row.get("image_id", "0"))
        return self.image_cache_dir / f"{image_id}.jpg"

    def _candidate_urls(self, row: Dict[str, Any]) -> List[str]:
        primary = str(row.get("image_url") or "").strip()
        if not primary:
            return []
        candidates = [primary]
        if "/VG_100K/" in primary:
            candidates.append(primary.replace("/VG_100K/", "/VG_100K_2/"))
        elif "/VG_100K_2/" in primary:
            candidates.append(primary.replace("/VG_100K_2/", "/VG_100K/"))
        deduped: List[str] = []
        seen = set()
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped

    def _download_if_missing(
        self, row: Dict[str, Any], path: Path, force: bool = False
    ) -> None:
        if not self.allow_url_download:
            if not path.exists():
                raise RuntimeError(f"Missing local image (downloads disabled): {path}")
            return
        if path.exists() and not force:
            return
        urls = self._candidate_urls(row)
        if not urls:
            raise RuntimeError(f"Missing image_url for image_id={row.get('image_id')}")
        last_error: Exception | None = None
        for url in urls:
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=30) as response:
                    image_bytes = response.read()
                with Image.open(BytesIO(image_bytes)) as probe:
                    probe.verify()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    handle.write(image_bytes)
                return
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                UnidentifiedImageError,
            ) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to download valid image for image_id={row.get('image_id')} from candidate URLs."
        ) from last_error

    def _open_local_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _load_image(self, row: Dict[str, Any]) -> Image.Image:
        image_path = self._resolve_image_path(row)
        self._download_if_missing(row, image_path)
        try:
            return self._open_local_image(image_path)
        except (UnidentifiedImageError, OSError):
            self.corrupt_image_count += 1
            if image_path.exists() and self.allow_url_download:
                image_path.unlink(missing_ok=True)
                try:
                    self._download_if_missing(row, image_path, force=True)
                    return self._open_local_image(image_path)
                except Exception:
                    pass
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _labels_to_multihot(self, labels: List[str]) -> torch.Tensor:
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        for label in labels:
            if label in self.label_vocab:
                target[self.label_vocab[label]] = 1.0
        return target

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        image = self._load_image(row)
        if "target" in row and "target_mask" in row:
            target = torch.tensor(row["target"], dtype=torch.float32)
            target_mask = torch.tensor(row["target_mask"], dtype=torch.float32)
            if target.numel() != self.num_labels:
                raise RuntimeError(
                    f"target length {target.numel()} != num_labels {self.num_labels} for row {idx}"
                )
        else:
            labels = row.get(self.positives_key, [])
            if not isinstance(labels, list):
                labels = []
            target = self._labels_to_multihot(labels)
            target_mask = torch.ones(self.num_labels, dtype=torch.float32)

        labels_out = row.get(self.positives_key, [])
        if not isinstance(labels_out, list):
            labels_out = []
        image_id = row.get("image_id")
        if image_id is None:
            image_id = 0
        return {
            "image": image,
            "target": target,
            "target_mask": target_mask,
            "image_id": int(image_id),
            "image_path": str(self._resolve_image_path(row)),
            "labels": labels_out,
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_samples,
    )


def class_pos_weights(
    label_frequencies: Dict[str, int],
    label_vocab: Dict[str, int],
) -> torch.Tensor:
    counts = torch.ones(len(label_vocab), dtype=torch.float32)
    for label, idx in label_vocab.items():
        counts[idx] = float(max(1, label_frequencies.get(label, 1)))
    weights = counts.max() / counts
    return weights


def split_paths(
    processed_root: Path,
    split_dir_name: str = "splits",
) -> Dict[str, Path]:
    split_dir = processed_root / split_dir_name
    return {
        "train": split_dir / "train.json",
        "val": split_dir / "val.json",
        "test": split_dir / "test.json",
    }

