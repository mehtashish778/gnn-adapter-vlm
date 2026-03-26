# Inference

## 1. Gradio UI (recommended)

Start:

```bash
cd /home/ashish/projects/GraphAttributeLearning/vlm_gnn_adapter
./.venv/bin/python app/gradio_xray_infer.py
```

The UI parameters:
- `Device`: `auto`, `cuda`, `cpu`
- `Top-K`: default `5` (1 to 30)
- `Threshold`: default `0.5` (probability cutoff)

## 2. Which checkpoint does the UI use?

The app loads a hardcoded checkpoint:

- `vlm_gnn_adapter/app/gradio_xray_infer.py`
  - `BEST_CKPT_PATH = REPO_ROOT / "outputs" / "full_xray_vlm_gnn_cuda1_bs4" / "best.pt"`

After you train a new run, either:
- overwrite that path to your run folder, or
- modify the app to accept a checkpoint path.

## 3. What comes out?

The app returns a table of:
- `label`
- `score(prob)` (sigmoid output)
- `positive@thr` (score >= threshold)

