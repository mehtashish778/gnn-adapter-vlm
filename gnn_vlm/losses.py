from __future__ import annotations

from typing import Optional

import torch


def bce_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE with logits; masked positions are ignored when mask is provided."""
    if mask is None:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return loss_fn(logits, targets)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none"
    )
    per = loss_fn(logits, targets)
    mask_f = mask.to(per.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (per * mask_f).sum() / denom

