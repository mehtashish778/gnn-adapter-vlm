from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_metrics(
    targets: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    per_attr_ap = []
    per_attr_f1 = []

    for idx in range(targets.shape[1]):
        if mask is not None:
            m = mask[:, idx] > 0.5
            yt = targets[m, idx]
            yp = preds[m, idx]
            ys = probs[m, idx]
        else:
            yt = targets[:, idx]
            yp = preds[:, idx]
            ys = probs[:, idx]
        per_attr_ap.append(_safe_average_precision(yt, ys))
        per_attr_f1.append(float(f1_score(yt, yp, zero_division=0)))

    if mask is not None:
        macro_f1 = float(
            f1_score(
                targets[mask > 0.5],
                preds[mask > 0.5],
                average="macro",
                zero_division=0,
            )
        )
        micro_f1 = float(
            f1_score(
                targets[mask > 0.5],
                preds[mask > 0.5],
                average="micro",
                zero_division=0,
            )
        )
    else:
        macro_f1 = float(
            f1_score(targets, preds, average="macro", zero_division=0)
        )
        micro_f1 = float(
            f1_score(targets, preds, average="micro", zero_division=0)
        )

    map_score = float(np.mean(per_attr_ap)) if per_attr_ap else 0.0
    return {
        "map": map_score,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_attribute_ap": per_attr_ap,
        "per_attribute_f1": per_attr_f1,
    }

