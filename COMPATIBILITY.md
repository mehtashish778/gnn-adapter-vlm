# Compatibility behavior

## 1. Legacy commands in the parent repo

The parent repo keeps compatibility wrappers:
- `scripts/train_gnn.py`
- `src/train/dataset.py`, `src/train/graph_builder.py`, `src/train/losses.py`, `src/train/metrics.py`, `src/train/qwen_vlm_encoders.py`

These wrappers delegate to the canonical implementation in:
- `vlm_gnn_adapter/gnn_vlm/`

So you can keep running the old parent commands, but the actual code path is the standalone repo.

## 2. Inference compatibility

The Gradio app is implemented in:
- `vlm_gnn_adapter/app/gradio_xray_infer.py`

It hardcodes `BEST_CKPT_PATH` unless you edit it:

```python
BEST_CKPT_PATH = REPO_ROOT / "outputs" / "full_xray_vlm_gnn_cuda1_bs4" / "best.pt"
```

To use your own trained adapter, change that constant to your run folder.

