from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from gnn_vlm.dataset import MultiLabelImageDataset, build_dataloader, load_json, split_paths
from gnn_vlm.metrics import compute_metrics
from gnn_vlm.qwen_vlm_encoders import FrozenQwen2VLEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot Qwen2-VL baseline (yes/no per label).")
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset_xray.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--snapshot-path", type=str, default="")
    parser.add_argument("--run-name", type=str, default="zero_shot_baseline")
    parser.add_argument("--question-template", type=str, default="Does this chest X-ray show {finding}? Answer yes or no.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def ordered_label_names(label_vocab: Dict[str, int]) -> List[str]:
    return [k for k, _ in sorted(label_vocab.items(), key=lambda kv: kv[1])]


def build_loader(dataset_cfg: Dict[str, Any], split: str, max_samples: int):
    processed_root = REPO_ROOT / dataset_cfg["dataset"]["processed_dir"]
    label_vocab_path = processed_root / dataset_cfg["processing"]["label_vocab_file"]
    splits = split_paths(processed_root=processed_root, split_dir_name=dataset_cfg["processing"]["split_dir"])
    split_path = splits[split]
    image_root = REPO_ROOT / dataset_cfg["dataset"]["root_dir"]

    ds = MultiLabelImageDataset(
        split_path=split_path,
        label_vocab_path=label_vocab_path,
        repo_root=REPO_ROOT,
        image_root=image_root,
        allow_url_download=False,
    )
    if max_samples > 0:
        ds.samples = ds.samples[:max_samples]
    loader = build_dataloader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(dataset_cfg["dataset"].get("num_workers", 0)),
        pin_memory=bool(dataset_cfg["dataset"].get("pin_memory", True)),
    )
    label_vocab = load_json(label_vocab_path)
    return loader, label_vocab


def build_yes_token_ids(tokenizer: Any) -> List[int]:
    candidates = [" yes", "yes", " Yes", "Yes"]
    ids: List[int] = []
    for text in candidates:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError("Could not resolve single-token ids for 'yes'.")
    return ids


def yes_probability_for_label(
    vlm: FrozenQwen2VLEncoder,
    pil_image: Any,
    label_name: str,
    question_template: str,
    yes_token_ids: List[int],
    device: torch.device,
) -> float:
    question = question_template.format(finding=label_name)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm.processor(text=[text], images=[pil_image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device=device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen = vlm.model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if not gen.scores:
        return 0.5
    scores = gen.scores[0][0].float()
    probs = torch.softmax(scores, dim=-1)
    prob_yes = float(probs[yes_token_ids].sum().item())
    return max(1e-6, min(1.0 - 1e-6, prob_yes))


def main() -> None:
    args = parse_args()
    dataset_cfg = load_yaml(args.dataset_config)
    device = resolve_device(args.device)
    loader, label_vocab = build_loader(dataset_cfg, args.split, args.max_samples)
    label_names = ordered_label_names(label_vocab)

    vlm = FrozenQwen2VLEncoder(
        repo_root=REPO_ROOT,
        model_id=args.model_id,
        snapshot_path=args.snapshot_path.strip() or None,
    )
    vlm.to(device)
    vlm.model.eval()
    yes_token_ids = build_yes_token_ids(vlm.processor.tokenizer)

    all_targets: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["images"][0]
            targets = batch["targets"][0].cpu().numpy().astype(np.float32)
            tmask = batch["target_mask"][0].cpu().numpy().astype(np.float32)
            image_path = str(batch.get("image_paths", [""])[0])

            probs = []
            for label in label_names:
                p_yes = yes_probability_for_label(
                    vlm=vlm,
                    pil_image=image,
                    label_name=label,
                    question_template=args.question_template,
                    yes_token_ids=yes_token_ids,
                    device=device,
                )
                probs.append(p_yes)
            probs_np = np.asarray(probs, dtype=np.float32)
            logits_np = np.log(probs_np / (1.0 - probs_np))

            all_targets.append(targets[None, :])
            all_masks.append(tmask[None, :])
            all_logits.append(logits_np[None, :])
            rows.append(
                {
                    "image_path": image_path,
                    "top5_probs": [
                        {
                            "label": label_names[int(i)],
                            "prob": float(probs_np[int(i)]),
                        }
                        for i in np.argsort(-probs_np)[:5]
                    ],
                }
            )

    targets_np = np.concatenate(all_targets, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    logits_np = np.concatenate(all_logits, axis=0)
    metrics = compute_metrics(targets=targets_np, logits=logits_np, threshold=0.5, mask=masks_np)

    output_dir = REPO_ROOT / "outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
    with (output_dir / "sample_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=True)
    print(json.dumps({"run_name": args.run_name, "map": metrics["map"]}, indent=2))


if __name__ == "__main__":
    main()

