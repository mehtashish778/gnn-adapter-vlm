from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from gnn_vlm.dataset import load_json
from gnn_vlm.qwen_vlm_encoders import FrozenQwen2VLEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hallucination ontology for CheXpert labels."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "xray",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "raw",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "xray" / "hallucination_ontology.json",
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--snapshot-path", type=str, default="")
    parser.add_argument("--prompt-template", type=str, default="chest X-ray showing {finding}")
    parser.add_argument(
        "--max-positives-per-label",
        type=int,
        default=128,
        help="Cap number of positive images encoded per label for visual overlap.",
    )
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=5,
        help="Number of clusters per label for visual overlap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def ordered_label_names(label_vocab: Dict[str, int]) -> List[str]:
    return [name for name, _ in sorted(label_vocab.items(), key=lambda x: x[1])]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def build_semantic_similarity_matrix(
    vlm: FrozenQwen2VLEncoder,
    label_names: List[str],
    prompt_template: str,
    device: torch.device,
) -> np.ndarray:
    prompts = [prompt_template.format(finding=name) for name in label_names]
    with torch.no_grad():
        z_txt = vlm.encode_texts(prompts, device=device)
        z_txt = F.normalize(z_txt, p=2, dim=-1)
        sim = z_txt @ z_txt.T
    return sim.detach().cpu().numpy()


def build_pmi_matrix(train_samples: List[Dict[str, Any]], num_labels: int) -> np.ndarray:
    counts = np.zeros(num_labels, dtype=np.float64)
    pair_counts = np.zeros((num_labels, num_labels), dtype=np.float64)
    total = 0.0
    eps = 1e-12

    for sample in train_samples:
        target = np.asarray(sample.get("target", []), dtype=np.float64)
        if target.size != num_labels:
            continue
        active = np.where(target > 0.5)[0]
        if active.size == 0:
            continue
        total += 1.0
        counts[active] += 1.0
        for i in active:
            pair_counts[i, active] += 1.0

    if total < 1.0:
        return np.zeros((num_labels, num_labels), dtype=np.float64)

    p = counts / total
    p2 = pair_counts / total
    denom = (p.reshape(-1, 1) * p.reshape(1, -1)) + eps
    pmi = np.log((p2 + eps) / denom)
    pmi = np.where(np.isfinite(pmi), pmi, 0.0)
    return pmi


def sample_positive_image_paths(
    train_samples: List[Dict[str, Any]],
    label_idx: int,
    image_root: Path,
    max_count: int,
) -> List[Path]:
    out: List[Path] = []
    for sample in train_samples:
        target = sample.get("target", [])
        if label_idx >= len(target) or float(target[label_idx]) <= 0.5:
            continue
        rel = str(sample.get("image_path", "")).strip()
        if not rel:
            continue
        full = image_root / rel.lstrip("/")
        if full.exists():
            out.append(full)
        if len(out) >= max_count:
            break
    return out


def encode_image_paths(
    vlm: FrozenQwen2VLEncoder,
    image_paths: List[Path],
    device: torch.device,
) -> np.ndarray:
    from PIL import Image

    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for path in image_paths:
            with Image.open(path) as img:
                pil = img.convert("RGB")
            emb = vlm.encode_images_pil([pil], device=device)[0]
            emb = F.normalize(emb, p=2, dim=-1)
            vectors.append(emb.detach().cpu().numpy())
    if not vectors:
        return np.zeros((0, vlm.vision_embed_dim), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)


def centroid_similarity(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    k: int,
    seed: int,
) -> float:
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        return 0.0
    ka = min(k, feats_a.shape[0])
    kb = min(k, feats_b.shape[0])
    kma = KMeans(n_clusters=ka, random_state=seed, n_init=10).fit(feats_a)
    kmb = KMeans(n_clusters=kb, random_state=seed, n_init=10).fit(feats_b)
    ca = torch.tensor(kma.cluster_centers_, dtype=torch.float32)
    cb = torch.tensor(kmb.cluster_centers_, dtype=torch.float32)
    ca = F.normalize(ca, p=2, dim=-1)
    cb = F.normalize(cb, p=2, dim=-1)
    sim = ca @ cb.T
    return float(sim.max().item())


def build_visual_overlap_matrix(
    vlm: FrozenQwen2VLEncoder,
    label_names: List[str],
    train_samples: List[Dict[str, Any]],
    image_root: Path,
    max_positives_per_label: int,
    kmeans_k: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    label_feats: List[np.ndarray] = []
    for i, _name in enumerate(label_names):
        paths = sample_positive_image_paths(
            train_samples=train_samples,
            label_idx=i,
            image_root=image_root,
            max_count=max_positives_per_label,
        )
        label_feats.append(encode_image_paths(vlm=vlm, image_paths=paths, device=device))

    n = len(label_names)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            value = centroid_similarity(label_feats[i], label_feats[j], kmeans_k, seed)
            out[i, j] = value
            out[j, i] = value
    return out


def to_pair_edges(
    label_names: List[str],
    sem: np.ndarray,
    pmi: np.ndarray,
    vis: np.ndarray,
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    n = len(label_names)
    for i in range(n):
        for j in range(i + 1, n):
            edges.append(
                {
                    "label_A": label_names[i],
                    "label_B": label_names[j],
                    "semantic_similarity": float(sem[i, j]),
                    "visual_cluster_overlap": float(vis[i, j]),
                    "statistical_pmi": float(pmi[i, j]),
                }
            )
    edges.sort(
        key=lambda r: (
            r["semantic_similarity"] + r["visual_cluster_overlap"] + max(0.0, r["statistical_pmi"])
        ),
        reverse=True,
    )
    return edges


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processed_dir = args.processed_dir
    split_path = processed_dir / "splits" / "train.json"
    label_vocab_path = processed_dir / "label_vocab.json"

    label_vocab = load_json(label_vocab_path)
    train_payload = load_json(split_path)
    train_samples = train_payload["samples"]
    label_names = ordered_label_names(label_vocab)
    num_labels = len(label_names)

    device = resolve_device(args.device)
    snap = args.snapshot_path.strip() or None
    vlm = FrozenQwen2VLEncoder(
        repo_root=REPO_ROOT,
        model_id=args.model_id,
        snapshot_path=snap,
    )
    vlm.to(device)

    sem = build_semantic_similarity_matrix(vlm, label_names, args.prompt_template, device)
    pmi = build_pmi_matrix(train_samples, num_labels)
    vis = build_visual_overlap_matrix(
        vlm=vlm,
        label_names=label_names,
        train_samples=train_samples,
        image_root=args.image_root,
        max_positives_per_label=args.max_positives_per_label,
        kmeans_k=args.kmeans_k,
        seed=args.seed,
        device=device,
    )

    payload = {
        "meta": {
            "model_id": args.model_id,
            "prompt_template": args.prompt_template,
            "num_labels": num_labels,
            "max_positives_per_label": args.max_positives_per_label,
            "kmeans_k": args.kmeans_k,
        },
        "labels": label_names,
        "edges": to_pair_edges(label_names, sem, pmi, vis),
        "semantic_sim_matrix": sem.tolist(),
        "pmi_matrix": pmi.tolist(),
        "visual_overlap_matrix": vis.tolist(),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(f"Wrote ontology: {args.output_path}")


if __name__ == "__main__":
    main()

