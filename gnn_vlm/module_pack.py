from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .dataset import load_json
from .graph_models import NativeGNNClassifier
from .qwen_vlm_encoders import FrozenQwen2VLEncoder, VLMAdapterStack


def _ordered_label_names(label_vocab: Dict[str, int]) -> List[str]:
    return [k for k, _ in sorted(label_vocab.items(), key=lambda kv: kv[1])]


def build_xray_vlm_modules(
    repo_root: Path,
    cfg: Dict[str, Any],
    num_labels: int,
    label_vocab: Dict[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    xv = cfg["xray_vlm"]
    snap = str(xv.get("snapshot_path") or "").strip() or None
    vlm = FrozenQwen2VLEncoder(
        repo_root,
        model_id=str(xv.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")),
        snapshot_path=snap,
        cache_dir=str(xv.get("cache_dir") or (repo_root / "data" / "hf_cache")),
    )
    vlm.to(device)
    gnn_dim = int(xv["gnn_dim"])
    adapters = VLMAdapterStack(vlm.vision_embed_dim, vlm.hidden_size, gnn_dim).to(device)

    names = _ordered_label_names(label_vocab)
    template = str(xv["prompt_template"])
    prompts = [template.format(finding=n) for n in names]
    with torch.no_grad():
        z_txt = vlm.encode_texts(prompts, device)
        attr_cached = adapters.proj_attr(z_txt)

    head = str(xv.get("head", "gnn"))
    train_edge_mode = str(xv.get("train_edge_mode", "positive_only"))
    eval_edge_mode = str(xv.get("eval_edge_mode", "full_bipartite"))
    ontology_payload: Dict[str, Any] | None = None
    if train_edge_mode == "ontology_weighted" or eval_edge_mode == "ontology_weighted":
        ontology_path = Path(
            str(
                xv.get("ontology_path")
                or (repo_root / "data" / "processed" / "xray" / "hallucination_ontology.json")
            )
        )
        if not ontology_path.is_absolute():
            ontology_path = repo_root / ontology_path
        if not ontology_path.exists():
            raise FileNotFoundError(
                f"ontology_weighted requested but ontology file not found: {ontology_path}"
            )
        ontology_payload = load_json(ontology_path)

    linear_head: nn.Linear | None = None
    gnn_model: NativeGNNClassifier | None = None
    if head == "linear":
        linear_head = nn.Linear(gnn_dim, num_labels).to(device)
    elif head == "gnn":
        hidden_dims = [layer_cfg["out_dim"] for layer_cfg in cfg["gnn"]["layers"]]
        dropout = float(cfg["gnn"].get("dropout", 0.2))
        gnn_model = NativeGNNClassifier(
            in_dim=gnn_dim,
            hidden_dims=hidden_dims,
            num_attributes=num_labels,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"xray_vlm.head must be gnn or linear, got {head}")

    return {
        "vlm": vlm,
        "adapters": adapters,
        "gnn_model": gnn_model,
        "linear_head": linear_head,
        "attr_cached": attr_cached,
        "head": head,
        "train_edge_mode": train_edge_mode,
        "eval_edge_mode": eval_edge_mode,
        "ontology": ontology_payload,
    }

