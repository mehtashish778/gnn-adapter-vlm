# VLM-GNN Adapter: Research Plan & Implementation Guide

## Overview

This document covers three interconnected research components built on top of the
existing `vlm_gnn_adapter` codebase (Frozen Qwen2-VL-2B + VLMAdapterStack +
NativeGNNClassifier on CheXpert 14-class multi-label classification).

**Central Thesis:** Domain adaptation of VLMs for medical imaging does not require
expensive SFT or RL. A small GNN adapter (~3-4M trainable params) attached at test
time to a frozen general-purpose VLM can match or approach the performance of
full medical VLMs (BioViL-T, CheXagent, LLaVA-Med).

---

## Part 1 — Hallucination Ontology for CLIP-Based VLMs

### Motivation

CLIP-derivative VLMs (including Qwen2-VL) learn text-image alignment from
co-occurrence statistics. This means the model cannot distinguish between:
- Labels that are visually similar (Consolidation vs. Edema on X-ray)
- Labels that are semantically close in LLM token space (Pneumonia vs. Consolidation)
- Labels that statistically co-occur in training reports (Edema + Pleural Effusion)

These three failure modes define three layers of the ontology.

---

### Three-Level Ontology Structure

#### Level 1 — Semantic Proximity (Text Embedding Space)

Encode all 14 CheXpert labels using the VLM's text encoder with the prompt
template `"chest X-ray showing {finding}"` (matching `configs/model_xray_vlm_gnn.yaml`).

```python
prompts = [f"chest X-ray showing {label}" for label in label_vocab]
z_txt = vlm.encode_texts(prompts, device)            # shape: [14, hidden_size]
sim_matrix = torch.nn.functional.cosine_similarity(
    z_txt.unsqueeze(1), z_txt.unsqueeze(0), dim=-1   # shape: [14, 14]
)
```

Pairs where cosine similarity > threshold (e.g. 0.85) are **semantic hallucination
risk pairs**. Expected clusters for CheXpert:

| Cluster | Labels | Risk Mechanism |
|---|---|---|
| Opacity / Infection | Lung Opacity, Consolidation, Pneumonia, Atelectasis, Edema | Overlapping radiologic terms in pretraining corpus |
| Pleural | Pleural Effusion, Pleural Other | Prefix similarity + co-occurrence |
| Cardiac / Mediastinal | Enlarged Cardiomediastinum, Cardiomegaly | Anatomically overlapping, near-identical report language |
| Structural outliers | Pneumothorax, Fracture, Support Devices, Lung Lesion | Visually/semantically distinct, low risk |
| Healthy | No Finding | Adversarial to all above; suppressed under uncertainty |

#### Level 2 — Visual Co-occurrence (Image Feature Space)

For each label, collect VLM vision features (`vlm.encode_images_pil`) of all
positive training examples. Cluster with k-means or UMAP. Label pairs whose
positive-example clusters overlap in vision feature space are **visual hallucination
attractors** — the pairs where bipartite GNN message passing does the heaviest lifting.

#### Level 3 — Statistical PMI (Training Data)

Build a label co-occurrence matrix from `data/processed/xray/splits/train.json`
targets. Edge weight between label A and label B:

```
PMI(A, B) = log[ P(A and B) / (P(A) * P(B)) ]
```

High PMI = model can be correct for the wrong reason (predicts both when only one
is present). High-risk pair example: Edema + Pleural Effusion (~40% co-occurrence
in CheXpert).

#### Ontology Edge Schema

```python
# Each edge in the ontology graph has three typed weights:
{
    "label_A": "Edema",
    "label_B": "Pleural Effusion",
    "semantic_similarity": 0.91,     # cosine sim in text embed space
    "visual_cluster_overlap": 0.74,  # IOU of positive-example clusters
    "statistical_pmi": 1.32          # from training label co-occurrence matrix
}
```

This ontology directly informs richer `edge_mode` strategies in `graph_builder.py`
beyond the current binary `positive_only` / `full_bipartite` modes.

---

## Part 2 — Qwen2-VL-2B Baseline Evaluation with Error Logging

### What the Current Code Does

`run_eval_xray()` in `gnn_vlm/train_main.py` collects aggregate metrics (mAP,
macro F1, micro F1) but discards per-sample detail. The `batch` dict does not
pass image paths through the collate function — only `images`, `targets`,
`target_mask` are returned.

### What to Build: `scripts/eval/eval_baseline_vlm.py`

A standalone eval script (no training) that:

1. Loads a **linear-head checkpoint** (`head = linear`, no GNN) — isolates raw
   VLM representation quality as Baseline-B.
2. Tracks image paths per sample by extending `MultiLabelImageDataset` to include
   `sample["image_path"]` in the returned dict.
3. Logs per-sample errors to `outputs/{run_name}/error_log.json`.

#### Error Log Entry Schema

```json
{
    "image_path": "data/raw/raw/valid/patient64589/study1/view1_frontal.jpg",
    "true_labels": ["Pleural Effusion", "Edema"],
    "predicted_positives": ["Pleural Effusion", "Consolidation"],
    "errors": {
        "false_positives": ["Consolidation"],
        "false_negatives": ["Edema"]
    },
    "top5_probs": [
        {"label": "Pleural Effusion", "prob": 0.87},
        {"label": "Consolidation",    "prob": 0.71},
        {"label": "Lung Opacity",     "prob": 0.65},
        {"label": "Edema",            "prob": 0.48},
        {"label": "Atelectasis",      "prob": 0.41}
    ]
}
```

Top-5 sorted probabilities are computed as:

```python
probs = torch.sigmoid(logits)                        # [bsz, 14]
top5_vals, top5_idx = probs.topk(5, dim=-1)          # [bsz, 5] each
```

#### Key Diagnostic Metric: Confusion Rank

For each false-negative label L, compute L's average predicted rank (position in
the sorted top-5 list). If Edema is consistently ranked #4-5 when it is the true
label, that is a **calibration failure** (not a representation failure) — the GNN
adapter can correct it without modifying the VLM weights at all.

### Zero-Shot Baseline (Baseline-A)

Use Qwen2-VL generation (not feature extraction) to classify via free-form prompts:

```
Prompt: "Does this chest X-ray show {finding}? Answer yes or no."
```

Call `model.generate()` for each of the 14 labels and parse the log-probability
of the "yes" token. This requires a separate `scripts/eval/zero_shot_eval.py`
distinct from the feature-extraction path in `gnn_vlm/qwen_vlm_encoders.py`.

---

## Part 3 — GNN Adapter at Test Time vs. Medical VLMs

### Existing Architecture (What Is Already Built)

```
Frozen Qwen2-VL-2B                              0 trainable params
    │
    ├── encode_images_pil() → [bsz, vision_embed_dim=1280]
    └── encode_texts()      → [num_labels, hidden_size=1536]
         │
VLMAdapterStack                                ~3.1M trainable params
    ├── object_proj: Linear(1280 → 512)
    └── attr_proj:   Linear(1536 → 512)
         │
NativeGNNClassifier                            ~0.8M trainable params
    ├── BipartiteMessagePassingLayer(512 → 512)
    ├── BipartiteMessagePassingLayer(512 → 256)
    └── Linear(256 → 14)
```

Total trainable: **~4M params** vs. 2B frozen VLM. Checkpoint size: ~12MB.
The base VLM stays completely unchanged and can serve other tasks simultaneously.

### Experiment Matrix

| ID | System | VLM | Adapter | Training Cost |
|---|---|---|---|---|
| A | Zero-shot | Qwen2-VL-2B (frozen) | None — generation prompting | 0 |
| B | Linear probe | Qwen2-VL-2B (frozen) | Linear head only (~3.6K params) | Minutes |
| C | **Ours (GNN)** | Qwen2-VL-2B (frozen) | VLMAdapterStack + NativeGNNClassifier | Hours |
| D | BioViL-T | microsoft/BioViL-T | Full model SFT on MIMIC-CXR | GPU-weeks |
| E | CheXagent | Stanford 7B | Full SFT on CheXpert | GPU-weeks |
| F | LLaVA-Med | LLaVA 7B | BioMed SFT | GPU-weeks |

### Hypothesis

System C (Ours) should outperform A and B substantially, and approach D/E/F on
CheXpert mAP, macro F1, and per-class AUC — specifically on the semantic-proximity
cluster (Lung Opacity, Consolidation, Pneumonia, Atelectasis, Edema) where
CLIP-based alignment fails hardest.

**Why it works:** The bipartite message passing in `NativeGNNClassifier` propagates
structured attribute information (text prompts per label) back to the image
representation at inference time. This gives the model label-structured context
that pure cosine similarity in CLIP space cannot provide, without touching VLM weights.

### Evaluation Protocol

All systems are evaluated on the **same CheXpert validation split**
(`data/processed/xray/splits/val.json`). Metrics are computed via `gnn_vlm/metrics.py`:

- mAP (primary ranking metric)
- Macro F1 @ threshold 0.5
- Micro F1 @ threshold 0.5
- Per-class AUC

For comparison models (D/E/F), wrap their inference to produce raw logits/probs
and pass them into the same `compute_metrics(targets_np, logits_np, mask_np)` call.
Since `compute_metrics` operates on raw numpy arrays, any model plugs in cleanly.

### Broader Implications

1. **SFT on VLMs is not always necessary for domain adaptation** — a small,
   cheap adapter trained on task-labeled data can substitute.
2. **The adapter is modular and portable** — ships as a ~12MB `.pt` file; the
   base VLM is unchanged.
3. **Data curation matters more than model size** — the ontology from Part 1 can
   guide better edge construction in the GNN (ontology-weighted edges), achieving
   further gains without increasing adapter parameter count.
4. **Generalizes across tasks** — retrain with any `label_vocab.json` and
   `prompt_template` to adapt to a new attribute vocabulary (e.g., pathology slides,
   dermoscopy, ophthalmology).

---

## File Reference Map

| Component | File |
|---|---|
| VLM encoder (frozen) | `gnn_vlm/qwen_vlm_encoders.py` — `FrozenQwen2VLEncoder` |
| Adapter projection layers | `gnn_vlm/qwen_vlm_encoders.py` — `VLMAdapterStack` |
| Bipartite GNN | `gnn_vlm/graph_models.py` — `NativeGNNClassifier` |
| Bipartite message passing | `gnn_vlm/graph_models.py` — `BipartiteMessagePassingLayer` |
| Graph construction | `gnn_vlm/graph_builder.py` — `build_bipartite_batch` |
| Module wiring | `gnn_vlm/module_pack.py` — `build_xray_vlm_modules` |
| Training loop | `gnn_vlm/train_main.py` — `train()` |
| Eval loop | `gnn_vlm/train_main.py` — `run_eval_xray()` |
| Metrics | `gnn_vlm/metrics.py` — `compute_metrics` |
| Dataset | `gnn_vlm/dataset.py` — `MultiLabelImageDataset` |
| Label vocabulary | `data/processed/xray/label_vocab.json` |
| Label frequencies | `data/processed/xray/label_frequencies.json` |
| GNN model config | `configs/model_xray_vlm_gnn.yaml` |
| Linear model config | `configs/model_xray_vlm_linear.yaml` |
| Training config | `configs/train_xray.yaml` |
| Dataset config | `configs/dataset_xray.yaml` |
| Inference UI | `app/gradio_xray_infer.py` |

---

## Next Implementation Steps

### Step 1 — Hallucination Ontology Builder

Add `scripts/analysis/build_hallucination_ontology.py`:

- Encode all 14 label prompts via `vlm.encode_texts()` → semantic similarity matrix
- Load training targets from `data/processed/xray/splits/train.json` → PMI matrix
- Encode positive training images per label → visual cluster overlap (k-means/UMAP)
- Output: `data/processed/xray/hallucination_ontology.json`

### Step 2 — Per-Sample Error Logger

Add `scripts/eval/eval_baseline_vlm.py`:

- Load linear-head checkpoint (Baseline-B)
- Extend dataset to return `image_path` per sample
- Run inference on validation/test split
- Write `outputs/{run_name}/error_log.json` (per-sample top-5 probs + error breakdown)
- Compute and print per-class confusion rank

### Step 3 — Zero-Shot VLM Eval

Add `scripts/eval/zero_shot_eval.py`:

- Use `model.generate()` for each of the 14 labels with a yes/no prompt
- Parse generation log-probs for "yes" token
- Feed into same `compute_metrics()` call
- Output: Baseline-A numbers for comparison table

### Step 4 — Medical VLM Comparison Wrapper

Add `scripts/eval/eval_medical_vlm.py`:

- Plug-in wrapper for BioViL-T (`microsoft/BioViL-T`) and/or CheXagent inference
- Same `compute_metrics()` call, same CheXpert val split
- Output: `outputs/comparison_table.json` with all system metrics side-by-side

### Step 5 — Ontology-Weighted Graph Builder

Extend `gnn_vlm/graph_builder.py` with a third edge mode:

```python
edge_mode: "ontology_weighted"
# Uses PMI / semantic_similarity as continuous edge weights
# instead of binary (0/1) positive_only presence
```

This is the bridge from the ontology (Part 1) into the GNN training (Part 3),
and is expected to improve mAP on the Opacity/Consolidation/Pneumonia cluster
specifically.
