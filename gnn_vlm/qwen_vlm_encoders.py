"""Frozen Qwen2-VL embeddings (local cache / offline)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Sequence

import torch
import torch.nn as nn

_HIDDEN = 1536


def _patch_broken_torchvision_for_hf() -> None:
    """Patch broken torchvision stubs so transformers does not hard-fail."""
    try:
        from torchvision.transforms import InterpolationMode  # noqa: F401

        return
    except ImportError:
        pass
    try:
        import transformers.utils as _tu
        import transformers.utils.import_utils as _iu
    except ImportError:
        return
    _fn = lambda: False
    _iu.is_torchvision_available = _fn  # type: ignore[assignment]
    _tu.is_torchvision_available = _fn  # type: ignore[assignment]
    v2 = getattr(_iu, "is_torchvision_v2_available", None)
    if v2 is not None and hasattr(v2, "cache_clear"):
        v2.cache_clear()


_patch_broken_torchvision_for_hf()


def _patch_qwen2vl_processor_video_dependency() -> None:
    """Skip Qwen2VL video processor auto-load when torchvision v2 is unavailable."""
    try:
        import torchvision.transforms.v2.functional as _tv2f  # noqa: F401

        return
    except Exception:
        pass
    try:
        from transformers import Qwen2VLProcessor
    except Exception:
        return
    original = Qwen2VLProcessor.get_attributes

    def _get_attrs_no_video(cls):  # type: ignore[override]
        attrs = list(original())
        return [a for a in attrs if a != "video_processor"]

    Qwen2VLProcessor.get_attributes = classmethod(_get_attrs_no_video)


_patch_qwen2vl_processor_video_dependency()


def _find_snapshot_under_cache(repo_root: Path, model_id: str) -> Path | None:
    safe = model_id.replace("/", "--")
    base = repo_root / "data" / "hf_cache" / f"models--{safe}"
    if not base.is_dir():
        return None
    snap = base / "snapshots"
    if not snap.is_dir():
        return None
    subs = sorted(p for p in snap.iterdir() if p.is_dir())
    return subs[0] if subs else None


class FrozenQwen2VLEncoder(nn.Module):
    """Image + text embedding extractor; vision forward one image at a time."""

    def __init__(
        self,
        repo_root: Path,
        *,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        snapshot_path: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.model_id = model_id
        try:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        except ImportError as exc:
            raise ImportError(
                "Install transformers>=4.45 for Qwen2-VL (see requirements.txt)."
            ) from exc

        if snapshot_path:
            local_path = Path(snapshot_path)
            if not local_path.is_dir():
                raise FileNotFoundError(f"Qwen snapshot not found: {local_path}")
        else:
            env_snap = os.environ.get("QWEN2_VL_LOCAL_SNAPSHOT", "").strip()
            if env_snap:
                local_path = Path(env_snap)
            else:
                found = _find_snapshot_under_cache(repo_root, model_id)
                if found is None:
                    raise FileNotFoundError(
                        f"No local snapshot under data/hf_cache for {model_id}. "
                        "Set xray_vlm.snapshot_path or QWEN2_VL_LOCAL_SNAPSHOT."
                    )
                local_path = found

        self._local_path = str(local_path)
        _ = cache_dir  # reserved for future hub cache override

        self.processor = Qwen2VLProcessor.from_pretrained(
            self._local_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        try:
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except Exception:
            use_bf16 = False
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._local_path,
            local_files_only=True,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._dtype = dtype

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size", _HIDDEN))

    @property
    def vision_embed_dim(self) -> int:
        vc = getattr(self.model.config, "vision_config", None)
        if vc is None:
            return 1280
        if hasattr(vc, "embed_dim"):
            return int(vc.embed_dim)
        return int(getattr(vc, "hidden_size", 1280))

    def encode_images_pil(self, images: Sequence[Any], device: torch.device) -> torch.Tensor:
        self.model.eval()
        out_list: List[torch.Tensor] = []
        root = self.model
        vis = getattr(root, "visual", None)
        if vis is None and hasattr(root, "model"):
            vis = getattr(root.model, "visual", None)
        if vis is None:
            raise AttributeError("Could not find vision tower on Qwen2VL model")
        with torch.no_grad():
            for pil in images:
                messages = [{"role": "user", "content": [{"type": "image", "image": pil}]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[pil],
                    padding=True,
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(
                    device=device, dtype=vis.get_dtype()
                )
                grid = inputs["image_grid_thw"].to(device=device)
                tok = vis(pixel_values, grid_thw=grid)
                if hasattr(tok, "last_hidden_state"):
                    hid = tok.last_hidden_state
                elif isinstance(tok, tuple):
                    hid = tok[0]
                else:
                    hid = tok
                pooled = hid.float().mean(dim=0)
                out_list.append(pooled)
        return torch.stack(out_list, dim=0).to(device=device)

    def encode_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            toks = self.processor.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            ids = toks["input_ids"].to(device=device)
            mask = toks["attention_mask"].to(device=device).unsqueeze(-1).float()
            embed_fn = self.model.get_input_embeddings()
            emb = embed_fn(ids).float()
            summed = (emb * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return pooled.to(device=device)


class VLMAdapterStack(nn.Module):
    """Project vision tokens and text prompts into a shared GNN dimension."""

    def __init__(self, object_vlm_dim: int, text_vlm_dim: int, gnn_dim: int) -> None:
        super().__init__()
        self.object_proj = nn.Linear(object_vlm_dim, gnn_dim)
        self.attr_proj = nn.Linear(text_vlm_dim, gnn_dim)

    def proj_object(self, x: torch.Tensor) -> torch.Tensor:
        return self.object_proj(x)

    def proj_attr(self, x: torch.Tensor) -> torch.Tensor:
        return self.attr_proj(x)

