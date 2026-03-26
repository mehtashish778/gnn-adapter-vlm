#!/usr/bin/env bash
# Compare frozen Qwen2-VL linear vs GNN on CheXpert smoke data.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
EVAL_CFG="${REPO}/configs/eval.yaml"
DS="${REPO}/configs/dataset_xray.yaml"
TR="${REPO}/configs/train_xray.yaml"

PY=python3
# Prefer the repo-level venv at GraphAttributeLearning/.venv.
if [[ -x "${REPO}/../.venv/bin/python" ]]; then
  PY="${REPO}/../.venv/bin/python"
elif [[ -x "${REPO}/.venv/bin/python" ]]; then
  PY="${REPO}/.venv/bin/python"
fi

"${PY}" -m gnn_vlm.train \
  --dataset-config "$DS" \
  --model-config "${REPO}/configs/model_xray_vlm_linear.yaml" \
  --train-config "$TR" \
  --eval-config "$EVAL_CFG" \
  --run-name smoke_xray_vlm_linear \
  --mode smoke \
  --device cuda

"${PY}" -m gnn_vlm.train \
  --dataset-config "$DS" \
  --model-config "${REPO}/configs/model_xray_vlm_gnn.yaml" \
  --train-config "$TR" \
  --eval-config "$EVAL_CFG" \
  --run-name smoke_xray_vlm_gnn \
  --mode smoke \
  --device cuda

echo "--- smoke linear ---"
cat "${REPO}/outputs/smoke_xray_vlm_linear/metrics.json" | head -c 800 || true
echo ""
echo "--- smoke gnn ---"
cat "${REPO}/outputs/smoke_xray_vlm_gnn/metrics.json" | head -c 800 || true

