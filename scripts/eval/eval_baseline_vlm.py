from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.io_utils import write_json  # noqa: E402
from train.config_utils import deep_merge, load_yaml  # noqa: E402

from gnn_vlm.dataset import MultiLabelImageDataset, build_dataloader, class_pos_weights, load_json, split_paths
from gnn_vlm.losses import bce_logits_loss
from gnn_vlm.metrics import compute_metrics
from gnn_vlm.module_pack import build_xray_vlm_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate linear baseline and log per-sample errors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset_xray.yaml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model_xray_vlm_linear.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs" / "train_xray.yaml")
    parser.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    parser.add_argument("--run-name", type=str, default="baseline_linear_eval")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def ordered_label_names(label_vocab: Dict[str, int]) -> List[str]:
    return [k for k, _ in sorted(label_vocab.items(), key=lambda kv: kv[1])]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    eval_cfg = load_yaml(args.eval_config)
    cfg = deep_merge(dataset_cfg, model_cfg)
    cfg = deep_merge(cfg, train_cfg)
    cfg = deep_merge(cfg, eval_cfg)
    return cfg


def build_eval_loader(cfg: Dict[str, Any], split: str):
    processed_root = REPO_ROOT / cfg["dataset"]["processed_dir"]
    label_vocab_path = processed_root / cfg["processing"]["label_vocab_file"]
    label_freq_path = processed_root / cfg["processing"]["label_frequency_file"]
    splits = split_paths(processed_root=processed_root, split_dir_name=cfg["processing"]["split_dir"])
    split_path = splits[split]

    label_vocab = load_json(label_vocab_path)
    label_freq = load_json(label_freq_path)
    image_root = REPO_ROOT / cfg["dataset"]["root_dir"]
    ds = MultiLabelImageDataset(
        split_path=split_path,
        label_vocab_path=label_vocab_path,
        repo_root=REPO_ROOT,
        image_root=image_root,
        allow_url_download=False,
    )
    loader = build_dataloader(
        dataset=ds,
        batch_size=int(cfg["batching"].get("eval_batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    return loader, label_vocab, label_freq


def topk_payload(probs_row: np.ndarray, label_names: List[str], k: int = 5) -> List[Dict[str, Any]]:
    k = min(k, probs_row.shape[0])
    idx = np.argsort(-probs_row)[:k]
    return [{"label": label_names[int(i)], "prob": float(probs_row[int(i)])} for i in idx]


def build_confusion_rank(
    probs: np.ndarray, targets: np.ndarray, threshold: float, label_names: List[str]
) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(np.int32)
    order = np.argsort(-probs, axis=1)
    ranks = {name: [] for name in label_names}
    for row in range(targets.shape[0]):
        for li, lname in enumerate(label_names):
            if targets[row, li] > 0.5 and preds[row, li] == 0:
                pos = int(np.where(order[row] == li)[0][0]) + 1
                ranks[lname].append(pos)
    out: Dict[str, Any] = {}
    for lname, vals in ranks.items():
        out[lname] = {
            "false_negative_count": int(len(vals)),
            "avg_rank_when_false_negative": float(np.mean(vals)) if vals else None,
        }
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    device = resolve_device(args.device)

    loader, label_vocab, label_freq = build_eval_loader(cfg, args.split)
    label_names = ordered_label_names(label_vocab)
    num_labels = len(label_names)

    pack = build_xray_vlm_modules(REPO_ROOT, cfg, num_labels=num_labels, label_vocab=label_vocab, device=device)
    if pack["head"] != "linear" or pack["linear_head"] is None:
        raise RuntimeError("Model config must set xray_vlm.head=linear for baseline evaluation.")

    ckpt = torch.load(args.checkpoint, map_location=device)
    pack["adapters"].load_state_dict(ckpt["adapters_state"])
    pack["linear_head"].load_state_dict(ckpt["linear_state"])
    pack["adapters"].eval()
    pack["linear_head"].eval()
    pack["vlm"].eval()

    pos_weight = class_pos_weights(label_frequencies=label_freq, label_vocab=label_vocab).to(device)

    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_paths: List[str] = []
    losses: List[float] = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            mask = batch["target_mask"].to(device)
            images = batch["images"]
            image_paths = batch.get("image_paths", [""] * len(images))

            z_img = pack["vlm"].encode_images_pil(images, device)
            h = pack["adapters"].proj_object(z_img)
            logits = pack["linear_head"](h)

            loss = bce_logits_loss(logits, targets, pos_weight=pos_weight, mask=mask)
            losses.append(float(loss.item()))

            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_masks.append(mask.detach().cpu().numpy())
            all_paths.extend([str(p) for p in image_paths])

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    preds_np = (probs_np >= args.threshold).astype(np.int32)

    metrics = compute_metrics(targets=targets_np, logits=logits_np, threshold=args.threshold, mask=masks_np)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    error_log: List[Dict[str, Any]] = []
    for i in range(probs_np.shape[0]):
        true_idx = np.where((targets_np[i] > 0.5) & (masks_np[i] > 0.5))[0].tolist()
        pred_idx = np.where((preds_np[i] > 0.5) & (masks_np[i] > 0.5))[0].tolist()
        false_pos = sorted(set(pred_idx) - set(true_idx))
        false_neg = sorted(set(true_idx) - set(pred_idx))
        error_log.append(
            {
                "image_path": all_paths[i] if i < len(all_paths) else "",
                "true_labels": [label_names[j] for j in true_idx],
                "predicted_positives": [label_names[j] for j in pred_idx],
                "errors": {
                    "false_positives": [label_names[j] for j in false_pos],
                    "false_negatives": [label_names[j] for j in false_neg],
                },
                "top5_probs": topk_payload(probs_np[i], label_names, k=5),
            }
        )

    confusion_rank = build_confusion_rank(
        probs=probs_np,
        targets=targets_np,
        threshold=args.threshold,
        label_names=label_names,
    )

    output_dir = REPO_ROOT / cfg.get("run", {}).get("output_dir", "outputs") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "error_log.json", error_log)
    write_json(output_dir / "confusion_rank.json", confusion_rank)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": args.split,
                "checkpoint": str(args.checkpoint),
                "threshold": args.threshold,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(json.dumps({"run_name": args.run_name, "map": metrics["map"]}, indent=2))


if __name__ == "__main__":
    main()

