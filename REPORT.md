# CheXpert Multi-Label Study Report

**Repository:** `vlm_gnn_adapter`  
**Task:** 14-class multi-label classification on CheXpert-style processed splits.  
**Backbone:** Frozen `Qwen2-VL-2B-Instruct` (vision + text embeddings); trainable adapter projections and optional linear or GNN head.

---

## 1. Objective

Evaluate whether a **small trainable adapter** on top of a **frozen general VLM** can outperform **zero-shot** prompting on chest X-ray findings, and whether a **bipartite GNN head** improves on a **linear probe** under the same frozen features.

---

## 2. Methods

### 2.1 Data and metrics

- **Labels:** 14 (from `data/processed/xray/label_vocab.json`).
- **Splits:** `data/processed/xray/splits/{train,val,test}.json`.
- **Metrics** (from `gnn_vlm/metrics.py`): mean Average Precision (**mAP**) over attributes, **macro F1**, **micro F1** at threshold **0.5** where applicable.

### 2.2 Model variants

| Variant | Description |
|--------|-------------|
| **Zero-shot** | Per-image, per-label yes/no via `model.generate()` (`scripts/eval/zero_shot_eval.py`). |
| **Linear probe** | Frozen VLM → `VLMAdapterStack` → `Linear(512 → 14)`; trained with BCE (`xray_vlm.head: linear`). |
| **GNN adapter** | Frozen VLM → adapters → `NativeGNNClassifier` on bipartite image–attribute graph (`xray_vlm.head: gnn`). |
| **Eval edge ablation (GNN)** | `eval_edge_mode: full_bipartite` vs `ontology_weighted` using `data/processed/xray/hallucination_ontology.json`. |

### 2.3 Training stability notes

- **DataLoader:** `num_workers: 0` recommended when training on GPU to avoid fork-after-CUDA **segmentation faults** (see `configs/dataset_xray.yaml` overrides in practice).
- **PyTorch / driver:** CUDA wheels must match the host driver (e.g. avoid `torch+cu130` on a driver that only supports through CUDA 12.4).

---

## 3. Results

### 3.1 Primary comparison (mAP)

| Run | Notes | Split | mAP | Macro F1 | Micro F1 |
|-----|--------|-------|-----|----------|----------|
| Zero-shot | `outputs/zero_shot_baseline` | val | **0.236** | 0.601 | 0.751 |
| Linear (eval script) | `outputs/baseline_linear_eval` | val | **0.374** | 0.588 | 0.699 |
| Linear (full train) | `outputs/full_xray_vlm_linear_cuda1_safe` | val (best) | **0.373** | 0.589 | 0.696 |
| Linear (full train) | same | **test** | **0.373** | 0.589 | 0.696 |
| GNN 3 epochs | `outputs/full_xray_vlm_gnn_cuda1_safe` | **test** | **0.219** | 0.467 | 0.511 |
| GNN 5 epochs, `full_bipartite` eval | `outputs/gnn_cuda0_eval_fullbip` | **test** | **0.219** | 0.420 | 0.433 |
| GNN 5 epochs, `ontology_weighted` eval | `outputs/gnn_cuda1_eval_ontology` | **test** | **0.220** | 0.407 | 0.456 |

**Raw JSON sources:** `outputs/*/metrics.json` (train runs nest final numbers under `test_metrics`).

### 3.2 Findings

1. **Supervised adapter strongly beats zero-shot on mAP:** linear probe **~0.37 mAP** vs zero-shot **~0.24 mAP** (val; linear test matches **~0.37** on the saved run).
2. **GNN head under current setup does not beat linear on mAP:** all GNN runs cluster near **~0.22 mAP** on test.
3. **`full_bipartite` vs `ontology_weighted` at eval** moved test mAP by only **~0.001** — not a meaningful difference in this configuration.
4. **Macro F1 is not aligned with mAP** (e.g. zero-shot has higher macro F1 than GNN but lower mAP), so **mAP should be the primary ranking metric** for this task.

---

## 4. Artifacts and scripts

| Artifact | Path |
|----------|------|
| Hallucination / label ontology (semantic, PMI, visual overlap) | `data/processed/xray/hallucination_ontology.json` |
| Ontology builder | `scripts/analysis/build_hallucination_ontology.py` |
| Linear eval + per-sample errors + top-5 probs | `scripts/eval/eval_baseline_vlm.py` |
| Zero-shot baseline | `scripts/eval/zero_shot_eval.py` |
| External medical VLM eval (Hugging Face) | `scripts/eval/eval_medical_vlm.py` |
| Metric aggregation helper | `scripts/eval/build_comparison_table.py` |
| Training entrypoint | `python -m gnn_vlm.train` |

---

## 5. Recommendations (next experiments)

1. **Fair comparison:** Evaluate linear and GNN on the **same split** in one pass (val and test) and fix threshold per model if needed (optional: tune threshold on val).
2. **GNN diagnostics:** Inspect calibration and per-class AP (several classes show **0 AP** in GNN `per_attribute_ap`); consider class weights, label smoothing, or longer training with tuned LR.
3. **Graph training:** Ablate `train_edge_mode` (e.g. `full_bipartite` during training) vs `positive_only`; current GNN may be under-connected or mis-matched to eval.
4. **Medical VLM baseline:** Run `eval_medical_vlm.py` (e.g. `microsoft/BioViL-T`) and merge into `outputs/comparison_table.json` for external reference.

---

## 6. Revision log

- **2026-03-30:** Report created from completed runs: zero-shot, linear (safe 3×epoch train + eval script), GNN (3×epoch + 5×epoch eval ablations).
