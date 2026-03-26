# Training

## 1. Canonical train command

Training runs from:

```bash
cd /home/ashish/projects/GraphAttributeLearning/vlm_gnn_adapter
./.venv/bin/python -m gnn_vlm.train \
  --dataset-config configs/dataset_xray.yaml \
  --model-config configs/model_xray_vlm_gnn.yaml \
  --train-config configs/train_xray.yaml \
  --eval-config configs/eval.yaml \
  --run-name <run_name> \
  --mode full \
  --device cuda
```

Use `configs/model_xray_vlm_linear.yaml` to train the linear-head baseline (same frozen VLM + adapters, different head).

## 2. Smoke vs full

The trainer has `--mode {smoke,full}`:
- `--mode smoke`
  - trains `1` epoch
  - caps dataset sizes (`train[:512]`, `val[:128]`, `test[:128]`)
  - clamps `num_workers` to at most `8`
  - also uses a small `batch_size=4` internally (independent of `configs/train_xray.yaml`)
- `--mode full`
  - uses `configs/train_xray.yaml` stage1 epochs and batch sizing

## 3. What actually trains?

The code builds:
- Frozen `Qwen2-VL` encoder (no gradients)
- Trainable `VLMAdapterStack` (always)
- Trainable head:
  - `gnn` head: `NativeGNNClassifier`
  - `linear` head: `torch.nn.Linear`

Checkpoint output:
- `outputs/<run-name>/best.pt`
- `outputs/<run-name>/metrics.json`

## 4. Edge modes and prompt template

From `configs/model_xray_vlm_*.yaml`:
- `xray_vlm.prompt_template: "chest X-ray showing {finding}"`
- `xray_vlm.train_edge_mode: positive_only`
- `xray_vlm.eval_edge_mode: full_bipartite`

## 5. Example: smoke run (linear vs gnn)

```bash
cd /home/ashish/projects/GraphAttributeLearning/vlm_gnn_adapter
./.venv/bin/python -m gnn_vlm.train \
  --dataset-config configs/dataset_xray.yaml \
  --model-config configs/model_xray_vlm_linear.yaml \
  --train-config configs/train_xray.yaml \
  --eval-config configs/eval.yaml \
  --run-name smoke_xray_vlm_linear \
  --mode smoke \
  --device cuda
```

```bash
./.venv/bin/python -m gnn_vlm.train \
  --dataset-config configs/dataset_xray.yaml \
  --model-config configs/model_xray_vlm_gnn.yaml \
  --train-config configs/train_xray.yaml \
  --eval-config configs/eval.yaml \
  --run-name smoke_xray_vlm_gnn \
  --mode smoke \
  --device cuda
```

