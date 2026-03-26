# Release / Materialization

This standalone repo can be “materialized” from the parent `GraphAttributeLearning/` repo by copying:
- `data/` (CheXpert raw + processed + `hf_cache/` for Qwen2-VL)
- `outputs/` (trained checkpoints)

## 1. Materialize scripts

Scripts are under:
- `scripts/materialize_data.sh`
- `scripts/materialize_outputs.sh`

Run them from the parent repo root:

```bash
cd /home/ashish/projects/GraphAttributeLearning
./vlm_gnn_adapter/scripts/materialize_data.sh
./vlm_gnn_adapter/scripts/materialize_outputs.sh
```

### `rsync` availability
If `rsync` is not installed, the scripts automatically fall back to `cp -a` (slower but functional).

## 2. Offline Qwen2-VL snapshot requirement

`gnn_vlm/qwen_vlm_encoders.py` loads Qwen2-VL using:
- `local_files_only=True`

So you must have `vlm_gnn_adapter/data/hf_cache/` populated for:
- `Qwen/Qwen2-VL-2B-Instruct`

If you want to override snapshot selection, use either:
- `configs/*` -> `xray_vlm.snapshot_path`
- environment variable `QWEN2_VL_LOCAL_SNAPSHOT`

## 3. Checkpoint format contract

The trainer saves `outputs/<run-name>/best.pt` with:
- `mode: xray_vlm`
- `label_vocab`
- `config` (the merged configs used to train)
- `adapters_state`
- (optional) `gnn_state`
- (optional) `linear_state`

The Gradio app expects the same structure.

