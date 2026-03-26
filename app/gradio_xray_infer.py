from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import gradio as gr
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gnn_vlm.graph_builder import build_bipartite_batch
from gnn_vlm.module_pack import build_xray_vlm_modules

BEST_CKPT_PATH = REPO_ROOT / "outputs" / "full_xray_vlm_gnn_cuda1_bs4" / "best.pt"


class XrayCheckpointCache:
    """Caches loaded xray_vlm packs to avoid reloading Qwen2-VL each request."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get(self, checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
        key = (str(checkpoint_path.resolve()), str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("mode") != "xray_vlm":
            raise ValueError(f"Unsupported checkpoint mode={checkpoint.get('mode')}; expected 'xray_vlm'.")

        label_vocab: Dict[str, int] = checkpoint["label_vocab"]
        cfg: Dict[str, Any] = checkpoint["config"]
        num_labels = len(label_vocab)

        pack = build_xray_vlm_modules(
            repo_root=REPO_ROOT,
            cfg=cfg,
            num_labels=num_labels,
            label_vocab=label_vocab,
            device=device,
        )

        pack["adapters"].load_state_dict(checkpoint["adapters_state"])
        if pack.get("gnn_model") is not None and checkpoint.get("gnn_state") is not None:
            pack["gnn_model"].load_state_dict(checkpoint["gnn_state"])
        if pack.get("linear_head") is not None and checkpoint.get("linear_state") is not None:
            pack["linear_head"].load_state_dict(checkpoint["linear_state"])

        pack["vlm"].eval()
        pack["adapters"].eval()
        if pack.get("gnn_model") is not None:
            pack["gnn_model"].eval()
        if pack.get("linear_head") is not None:
            pack["linear_head"].eval()

        inv_vocab = {idx: label for label, idx in label_vocab.items()}
        cached = {"pack": pack, "label_vocab": label_vocab, "inv_vocab": inv_vocab}
        self._cache[key] = cached
        return cached


CHECKPOINT_CACHE = XrayCheckpointCache()


def _resolve_device(device_choice: str) -> torch.device:
    if device_choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_xray_vlm(
    image: Image.Image | None,
    model_device: str,
    top_k: int,
    threshold: float,
) -> Tuple[List[List[Any]], str]:
    if image is None:
        return [], "No image provided."

    checkpoint_path = BEST_CKPT_PATH
    if not checkpoint_path.is_file():
        return [], f"Checkpoint not found: {checkpoint_path}"

    device = _resolve_device(model_device)
    try:
        loaded = CHECKPOINT_CACHE.get(checkpoint_path, device=device)
        pack: Dict[str, Any] = loaded["pack"]
        inv_vocab: Dict[int, str] = loaded["inv_vocab"]
        num_labels = len(loaded["label_vocab"])

        with torch.no_grad():
            z_img = pack["vlm"].encode_images_pil([image], device)
            h_obj = pack["adapters"].proj_object(z_img)  # [1, gnn_dim]

            if pack["head"] == "linear":
                assert pack["linear_head"] is not None
                logits = pack["linear_head"](h_obj)  # [1, num_labels]
            else:
                assert pack["gnn_model"] is not None
                obj = h_obj.unsqueeze(1)  # [1, 1, gnn_dim]
                attr = pack["attr_cached"].unsqueeze(0).expand(1, -1, -1)  # [1, num_labels, gnn_dim]
                dummy_targets = torch.zeros(1, num_labels, device=device)
                graph = build_bipartite_batch(
                    feats=obj,
                    targets=dummy_targets,
                    attr_feats=attr,
                    edge_mode=str(pack["eval_edge_mode"]),
                )
                logits = pack["gnn_model"](graph)  # [1, num_labels]

            probs = logits.sigmoid().detach().cpu().view(-1)
            k = min(int(top_k), int(num_labels))
            values, indices = torch.topk(probs, k=k)

            rows: List[List[Any]] = []
            positives: List[str] = []
            for v, idx in zip(values.tolist(), indices.tolist()):
                label = inv_vocab[int(idx)]
                is_pos = float(v) >= float(threshold)
                if is_pos:
                    positives.append(label)
                rows.append([label, round(float(v), 6), is_pos])

        msg = f"Positives (>= {threshold:.2f}): {', '.join(positives) if positives else 'none'}"
        return rows, msg
    except Exception as exc:  # noqa: BLE001
        return [], f"Error: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("# Chest X-ray (CheXpert) — Frozen VLM + Adapters + GNN inference")
    gr.Markdown(
        "Using checkpoint: "
        f"`{BEST_CKPT_PATH}`\n\n"
        "Upload an X-ray image to get top predicted findings."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input X-ray image")
            device_choice = gr.Radio(choices=["auto", "cuda", "cpu"], value="auto", label="Device")
            top_k = gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Top-K")
            threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Threshold")
            run_btn = gr.Button("Run inference")

        with gr.Column(scale=2):
            table = gr.Dataframe(
                headers=["label", "score(prob)", "positive@thr"],
                datatype=["str", "number", "bool"],
                row_count=(0, "dynamic"),
                col_count=(3, 3),
                label="Predicted findings",
            )
            message = gr.Markdown()

    run_btn.click(
        infer_xray_vlm,
        inputs=[image_input, device_choice, top_k, threshold],
        outputs=[table, message],
    )


if __name__ == "__main__":
    demo.launch()

