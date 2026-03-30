from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from gnn_vlm.dataset import MultiLabelImageDataset, build_dataloader, load_json, split_paths
from gnn_vlm.metrics import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medical VLMs on CheXpert labels.")
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset_xray.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model-name", type=str, required=True, help="HF model id, e.g. microsoft/BioViL-T")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--prompt-template", type=str, default="chest X-ray showing {finding}")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split.")
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


def _encode_text_features(
    model: Any,
    processor: Any,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    if hasattr(model, "get_text_features"):
        text_feats = model.get_text_features(**text_inputs)
    else:
        out = model(**text_inputs)
        if hasattr(out, "text_embeds"):
            text_feats = out.text_embeds
        elif hasattr(out, "last_hidden_state"):
            text_feats = out.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Unable to extract text features from model output.")
    return F.normalize(text_feats.float(), p=2, dim=-1)


def _encode_image_features(
    model: Any,
    processor: Any,
    image: Any,
    device: torch.device,
) -> torch.Tensor:
    image_inputs = processor(images=[image], return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    if hasattr(model, "get_image_features"):
        image_feats = model.get_image_features(**image_inputs)
    else:
        out = model(**image_inputs)
        if hasattr(out, "image_embeds"):
            image_feats = out.image_embeds
        elif hasattr(out, "last_hidden_state"):
            image_feats = out.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Unable to extract image features from model output.")
    return F.normalize(image_feats.float(), p=2, dim=-1)


def evaluate_model(
    model: Any,
    processor: Any,
    loader: Any,
    label_names: List[str],
    prompt_template: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompts = [prompt_template.format(finding=label) for label in label_names]
    with torch.no_grad():
        text_feats = _encode_text_features(model, processor, prompts, device)  # [num_labels, d]

    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["images"][0]
            image_feats = _encode_image_features(model, processor, image, device)  # [1, d]
            logits = image_feats @ text_feats.T  # [1, num_labels]

            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(batch["targets"].numpy())
            all_masks.append(batch["target_mask"].numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    return logits_np, targets_np, masks_np


def main() -> None:
    args = parse_args()
    dataset_cfg = load_yaml(args.dataset_config)
    device = resolve_device(args.device)
    run_name = args.run_name.strip() or args.model_name.replace("/", "__")

    loader, label_vocab = build_loader(dataset_cfg, args.split, args.max_samples)
    label_names = ordered_label_names(label_vocab)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    logits_np, targets_np, masks_np = evaluate_model(
        model=model,
        processor=processor,
        loader=loader,
        label_names=label_names,
        prompt_template=args.prompt_template,
        device=device,
    )
    metrics = compute_metrics(targets=targets_np, logits=logits_np, threshold=0.5, mask=masks_np)

    output_dir = REPO_ROOT / "outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "split": args.split,
                "prompt_template": args.prompt_template,
                "max_samples": args.max_samples,
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    print(json.dumps({"run_name": run_name, "map": metrics["map"]}, indent=2))


if __name__ == "__main__":
    main()

