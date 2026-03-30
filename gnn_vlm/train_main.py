from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import write_json  # noqa: E402
from train.config_utils import (  # noqa: E402
    apply_dot_overrides,
    deep_merge,
    load_yaml,
    parse_overrides_json,
)  # noqa: E402

from .dataset import (  # noqa: E402
    MultiLabelImageDataset,
    build_dataloader,
    class_pos_weights,
    load_json,
    split_paths,
)
from .graph_builder import build_bipartite_batch  # noqa: E402
from .graph_models import NativeGNNClassifier  # noqa: E402
from .losses import bce_logits_loss  # noqa: E402
from .metrics import compute_metrics  # noqa: E402
from .module_pack import build_xray_vlm_modules  # noqa: E402


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [train_xray_vlm] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xray_vlm adapters + gnn/linear head.")
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--overrides-json", type=str, default="")
    parser.add_argument("--mode", type=str, default="full", choices=["smoke", "full"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu, or cuda:N (e.g. cuda:1)",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def load_combined_config(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    eval_cfg = load_yaml(args.eval_config)
    combined = deep_merge(dataset_cfg, model_cfg)
    combined = deep_merge(combined, train_cfg)
    combined = deep_merge(combined, eval_cfg)
    overrides = parse_overrides_json(args.overrides_json)
    if overrides:
        combined = apply_dot_overrides(combined, overrides)
    return combined


def build_loaders(cfg: Dict[str, Any], batch_size: int, eval_batch_size: int, smoke: bool):
    processed_root = REPO_ROOT / cfg["dataset"]["processed_dir"]
    label_vocab_path = processed_root / cfg["processing"]["label_vocab_file"]
    label_freq_path = processed_root / cfg["processing"]["label_frequency_file"]
    splits = split_paths(processed_root=processed_root, split_dir_name=cfg["processing"]["split_dir"])

    label_vocab = load_json(label_vocab_path)
    label_freq = load_json(label_freq_path)

    loader_type = str(cfg["dataset"].get("loader", "multi_label_local"))
    if loader_type != "multi_label_local":
        raise ValueError(f"Only loader=multi_label_local supported in xray standalone. Got loader_type={loader_type}")

    image_root = REPO_ROOT / cfg["dataset"]["root_dir"]
    train_ds = MultiLabelImageDataset(
        splits["train"],
        label_vocab_path,
        REPO_ROOT,
        image_root,
        allow_url_download=False,
    )
    val_ds = MultiLabelImageDataset(
        splits["val"],
        label_vocab_path,
        REPO_ROOT,
        image_root,
        allow_url_download=False,
    )
    test_ds = MultiLabelImageDataset(
        splits["test"],
        label_vocab_path,
        REPO_ROOT,
        image_root,
        allow_url_download=False,
    )

    if smoke:
        train_ds.samples = train_ds.samples[:512]
        val_ds.samples = val_ds.samples[:128]
        test_ds.samples = test_ds.samples[:128]

    num_workers = int(cfg["dataset"].get("num_workers", 0))
    if smoke:
        num_workers = min(num_workers, 8)

    train_loader = build_dataloader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    val_loader = build_dataloader(
        dataset=val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    test_loader = build_dataloader(
        dataset=test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    return train_loader, val_loader, test_loader, {"vocab": label_vocab, "freq": label_freq}


def run_eval_xray(
    pack: Dict[str, Any],
    loader: Any,
    device: torch.device,
    pos_weight: torch.Tensor,
    edge_mode: str,
) -> Dict[str, Any]:
    vlm = pack["vlm"]
    adapters: nn.Module = pack["adapters"]
    gnn_model: NativeGNNClassifier | None = pack["gnn_model"]
    linear_head: nn.Linear | None = pack["linear_head"]
    attr_cached: torch.Tensor = pack["attr_cached"]
    head = pack["head"]

    vlm.eval()
    adapters.eval()
    if gnn_model is not None:
        gnn_model.eval()
    if linear_head is not None:
        linear_head.eval()

    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    losses: List[float] = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            mask = batch["target_mask"].to(device) if "target_mask" in batch else torch.ones_like(targets)
            images = batch["images"]

            z_img = vlm.encode_images_pil(images, device)
            h = adapters.proj_object(z_img)

            if head == "linear":
                assert linear_head is not None
                logits = linear_head(h)
            else:
                assert gnn_model is not None
                obj = h.unsqueeze(1)
                bsz = obj.shape[0]
                attr = attr_cached.unsqueeze(0).expand(bsz, -1, -1)
                graph = build_bipartite_batch(
                    feats=obj,
                    targets=targets,
                    attr_feats=attr,
                    edge_mode=edge_mode,
                    ontology=pack.get("ontology"),
                )
                logits = gnn_model(graph)

            loss = bce_logits_loss(logits, targets, pos_weight=pos_weight, mask=mask)
            losses.append(float(loss.item()))
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_masks.append(mask.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 1))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 1))
    masks_np = np.concatenate(all_masks, axis=0) if all_masks else None
    metrics = compute_metrics(targets=targets_np, logits=logits_np, mask=masks_np)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train() -> None:
    args = parse_args()
    log(f"Starting run_name={args.run_name} mode={args.mode} device_request={args.device}")

    cfg = load_combined_config(args)
    run_cfg = cfg.get("run", {})
    seed = int(run_cfg.get("seed", 42))
    deterministic = bool(run_cfg.get("deterministic", True))
    set_seed(seed=seed, deterministic=deterministic)

    smoke = args.mode == "smoke"
    train_epochs = 1 if smoke else int(cfg["training"]["stage1"].get("epochs", 20))
    batch_size = 4 if smoke else int(cfg["batching"].get("batch_size", 32))
    eval_batch_size = 4 if smoke else int(cfg["batching"].get("eval_batch_size", 64))

    cuda_available = torch.cuda.is_available()
    if args.device == "auto":
        selected_device = "cuda" if cuda_available else "cpu"
    else:
        selected_device = args.device
    if selected_device.startswith("cuda") and not cuda_available:
        raise RuntimeError("CUDA requested but not available in environment.")
    device = torch.device(selected_device)

    train_loader, val_loader, test_loader, label_info = build_loaders(cfg, batch_size, eval_batch_size, smoke)
    label_vocab = label_info["vocab"]
    label_freq = label_info["freq"]
    num_labels = len(label_vocab)
    log(f"Label space size: {num_labels}")

    xray_enabled = bool(cfg.get("xray_vlm", {}).get("enabled", False))
    if not xray_enabled:
        raise ValueError("This standalone trainer only supports cfg.xray_vlm.enabled=true")

    log("Building frozen Qwen2-VL encoder + adapters (local cache)...")
    pack = build_xray_vlm_modules(REPO_ROOT, cfg, num_labels=num_labels, label_vocab=label_vocab, device=device)
    log(f"xray_vlm head={pack['head']} train_edges={pack['train_edge_mode']} eval_edges={pack['eval_edge_mode']}")

    train_params: List[nn.Parameter] = list(pack["adapters"].parameters())
    if pack["gnn_model"] is not None:
        train_params += list(pack["gnn_model"].parameters())
    if pack["linear_head"] is not None:
        train_params += list(pack["linear_head"].parameters())

    pos_weight = class_pos_weights(label_frequencies=label_freq, label_vocab=label_vocab).to(device)

    lr = float(cfg["training"]["stage1"].get("lr_head", 1e-3))
    weight_decay = float(cfg["optimization"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)

    output_root = REPO_ROOT / run_cfg.get("output_dir", "outputs")
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_map = -1.0
    best_path = run_dir / "best.pt"
    grad_clip = float(cfg["optimization"].get("grad_clip_norm", 1.0))
    ograd_accum_steps = max(1, int(cfg["optimization"].get("gradient_accumulation_steps", 1)))

    for epoch in range(1, train_epochs + 1):
        step_times: List[float] = []
        start_epoch = time.time()
        train_losses: List[float] = []

        optimizer.zero_grad(set_to_none=True)
        pack["vlm"].eval()
        pack["adapters"].train()
        if pack["gnn_model"] is not None:
            pack["gnn_model"].train()
        if pack["linear_head"] is not None:
            pack["linear_head"].train()

        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{train_epochs}")):
            batch_start = time.time()
            targets = batch["targets"].to(device)
            tmask = batch["target_mask"].to(device)
            images = batch["images"]

            z_img = pack["vlm"].encode_images_pil(images, device)
            h_obj = pack["adapters"].proj_object(z_img)

            if pack["head"] == "linear":
                assert pack["linear_head"] is not None
                logits = pack["linear_head"](h_obj)
            else:
                assert pack["gnn_model"] is not None
                obj = h_obj.unsqueeze(1)
                bsz = obj.shape[0]
                attr = pack["attr_cached"].unsqueeze(0).expand(bsz, -1, -1)
                graph = build_bipartite_batch(
                    feats=obj,
                    targets=targets,
                    attr_feats=attr,
                    edge_mode=pack["train_edge_mode"],
                    ontology=pack.get("ontology"),
                )
                logits = pack["gnn_model"](graph)

            loss = bce_logits_loss(logits, targets, pos_weight=pos_weight, mask=tmask)

            loss_scaled = loss / ograd_accum_steps
            loss_scaled.backward()

            if (step_idx + 1) % ograd_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(train_params, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(float(loss.item()))
            step_times.append(time.time() - batch_start)

        epoch_time = time.time() - start_epoch
        log(f"Epoch {epoch} finished in {epoch_time:.1f}s, avg_step_time={float(np.mean(step_times)) if step_times else 0.0:.3f}s, steps={len(step_times)}")

        val_metrics = run_eval_xray(
            pack=pack,
            loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            edge_mode=pack["eval_edge_mode"],
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_map": val_metrics["map"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
        }
        history.append(epoch_row)
        log(f"Epoch {epoch}/{train_epochs} -> train_loss={epoch_row['train_loss']:.4f} val_map={epoch_row['val_map']:.4f} val_macro_f1={epoch_row['val_macro_f1']:.4f}")

        if val_metrics["map"] > best_map:
            best_map = val_metrics["map"]
            ckpt: Dict[str, Any] = {"label_vocab": label_vocab, "config": cfg, "mode": "xray_vlm"}
            ckpt["adapters_state"] = pack["adapters"].state_dict()
            ckpt["gnn_state"] = pack["gnn_model"].state_dict() if pack.get("gnn_model") is not None else None
            ckpt["linear_state"] = pack["linear_head"].state_dict() if pack.get("linear_head") is not None else None
            torch.save(ckpt, best_path)
            log(f"Saved new best checkpoint at epoch {epoch}: {best_path}")

    log("Running final test evaluation using best checkpoint...")
    checkpoint = torch.load(best_path, map_location=device)
    pack["adapters"].load_state_dict(checkpoint["adapters_state"])
    if checkpoint.get("gnn_state") is not None and pack.get("gnn_model") is not None:
        pack["gnn_model"].load_state_dict(checkpoint["gnn_state"])
    if checkpoint.get("linear_state") is not None and pack.get("linear_head") is not None:
        pack["linear_head"].load_state_dict(checkpoint["linear_state"])

    test_metrics = run_eval_xray(
        pack=pack,
        loader=test_loader,
        device=device,
        pos_weight=pos_weight,
        edge_mode=pack["eval_edge_mode"],
    )

    payload = {
        "run_name": args.run_name,
        "mode": args.mode,
        "num_labels": num_labels,
        "best_val_map": best_map,
        "history": history,
        "test_metrics": test_metrics,
        "xray_vlm": True,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "label_vocab.json", label_vocab)
    with (run_dir / "overrides.json").open("w", encoding="utf-8") as handle:
        json.dump(parse_overrides_json(args.overrides_json), handle, indent=2, ensure_ascii=True)

    log(f"Training complete for {args.run_name}")
    print(json.dumps({"run_name": args.run_name, "test_map": test_metrics["map"]}, indent=2), flush=True)


def main() -> None:
    train()


if __name__ == "__main__":
    main()

