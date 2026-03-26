# VLM->Adapters->GNN X-ray Adapter (standalone)

This folder is a standalone, xray-focused extract of the project.

## Quick start

1. Prepare data (CheXpert)
   - Put CheXpert CSVs + image folders under `data/raw/raw/` (see `scripts/data/process_chexpert.py` for exact expectations).
   - Run:
     - `python scripts/data/process_chexpert.py --raw-root data/raw/raw --out-dir data/processed/xray`
2. Smoke train:
   - `python -m gnn_vlm.train --dataset-config configs/dataset_xray.yaml --model-config configs/model_xray_vlm_gnn.yaml --train-config configs/train_xray.yaml --eval-config configs/eval.yaml --run-name smoke_xray_vlm_gnn --mode smoke --device cuda`
3. Inference (Gradio):
   - `python app/gradio_xray_infer.py`

## Environment
- Use the repo venv (system Python may not have `torch`): `./.venv/bin/python -m gnn_vlm.train ...`
- Data/output materialization:
  - `scripts/materialize_data.sh` (copies `data/raw/raw`, `data/processed/xray`, `data/hf_cache`)
  - `scripts/materialize_outputs.sh` (copies xray run folders from `outputs/`)
  - If `rsync` isn’t installed, the scripts fall back to `cp -a` (slower, but functional).

## Full Docs
- Setup: `SETUP.md`
- Data prep: `DATA.md`
- Training: `TRAINING.md`
- Inference (Gradio): `INFERENCE.md`
- Release / materialization: `RELEASE.md`
- Compatibility shims: `COMPATIBILITY.md`

## Expected directory layout

- `data/raw/raw/` - CheXpert CSVs + images
- `data/processed/xray/` - generated `label_vocab.json`, `label_frequencies.json`, `splits/*.json`
- `data/hf_cache/` - local HuggingFace snapshots for `Qwen2-VL` (offline use)
- `outputs/` - checkpoints saved by training
- `configs/` - dataset/model/train/eval YAMLs
- `gnn_vlm/` - the canonical implementation code

## Notes

- The trainer supports two heads via `configs/model_xray_vlm_gnn.yaml` / `configs/model_xray_vlm_linear.yaml`:
  - `xray_vlm.head: gnn`
  - `xray_vlm.head: linear`

