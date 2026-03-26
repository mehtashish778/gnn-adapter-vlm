# Setup

## 1. Python environment

This repo expects a working PyTorch install. Use the repo venv at `GraphAttributeLearning/.venv`.

If you need to (re)install dependencies:

```bash
cd /home/ashish/projects/GraphAttributeLearning
./.venv/bin/python -m pip install -r vlm_gnn_adapter/requirements.txt
```

Notes:
- `vlm_gnn_adapter/requirements.txt` includes `torch`, `torchvision`, `transformers`, and `gradio`.
- The training code uses offline Qwen2-VL snapshots (`local_files_only=True`), so you must also have `vlm_gnn_adapter/data/hf_cache/` populated (see `RELEASE.md`).

## 2. Canonical entrypoint

Training is run via the module entrypoint:

```bash
./.venv/bin/python -m gnn_vlm.train --help
```

The legacy wrapper in the parent repo (`scripts/train_gnn.py`) delegates to the same implementation.

