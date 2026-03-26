# Data (CheXpert)

## 1. Expected input layout (raw)

The trainer assumes this dataset config:
- `configs/dataset_xray.yaml`
  - `dataset.root_dir: data/raw/raw`
  - `dataset.processed_dir: data/processed/xray`
  - splits live under `data/processed/xray/splits/`

So you should place CheXpert in:

- `vlm_gnn_adapter/data/raw/raw/train.csv`
- `vlm_gnn_adapter/data/raw/raw/valid.csv`
- `vlm_gnn_adapter/data/raw/raw/train/` (images referenced by `Path` in `train.csv`)
- `vlm_gnn_adapter/data/raw/raw/valid/` (images referenced by `Path` in `valid.csv`)

The `Path` column is expected to include the prefix `CheXpert-v1.0-small/` by default; the script strips it.

## 2. Generate processed manifests

Run:

```bash
cd /home/ashish/projects/GraphAttributeLearning/vlm_gnn_adapter
./.venv/bin/python scripts/data/process_chexpert.py \
  --raw-root data/raw/raw \
  --out-dir data/processed/xray
```

Outputs created under `data/processed/xray/`:
- `label_vocab.json` (label -> index)
- `label_frequencies.json` (label counts used for `pos_weight`)
- `splits/train.json`
- `splits/val.json`
- `splits/test.json` (defaults to a copy of val)
- `processing_report.json`

## 3. Script options (important)

`scripts/data/process_chexpert.py` supports:
- `--strip-prefix` (default: `CheXpert-v1.0-small/`)
- `--uncertain-handling` (default: `mask`; choices: `mask|positive|negative`)
- `--omit-unmentioned` (if set, rows with empty cells are masked out instead of treated as negatives)
- `--no-test-from-valid` (if set, test split is empty)

