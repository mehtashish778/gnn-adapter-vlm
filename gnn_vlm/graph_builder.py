from __future__ import annotations

from typing import List, Optional

import torch

from .graph_models import BipartiteGraphBatch


def build_bipartite_batch(
    feats: torch.Tensor,
    targets: torch.Tensor,
    attr_feats: Optional[torch.Tensor] = None,
    edge_mode: str = "positive_only",
) -> BipartiteGraphBatch:
    """Construct a bipartite batch graph from object features and multi-hot targets."""
    device = feats.device
    bsz, num_objects, dim_o = feats.shape
    num_attrs = targets.shape[1]

    if attr_feats is None:
        attr_tensor = torch.zeros(bsz, num_attrs, dim_o, device=device)
    else:
        attr_tensor = attr_feats.to(device=device, dtype=feats.dtype)
        if attr_tensor.dim() == 2:
            attr_tensor = attr_tensor.unsqueeze(0).expand(bsz, -1, -1)
        if attr_tensor.shape[0] != bsz or attr_tensor.shape[1] != num_attrs:
            raise ValueError(
                f"attr_feats shape {attr_tensor.shape} incompatible with batch {bsz} attrs {num_attrs}"
            )
        if attr_tensor.shape[-1] != dim_o:
            raise ValueError(
                f"attr_feats last dim {attr_tensor.shape[-1]} != object feat dim {dim_o}"
            )

    object_indices: List[int] = []
    attr_indices: List[int] = []

    for b in range(bsz):
        for o in range(num_objects):
            flat_obj_idx = b * num_objects + o
            if edge_mode == "full_bipartite":
                for a in range(num_attrs):
                    object_indices.append(flat_obj_idx)
                    attr_indices.append(b * num_attrs + a)
            elif edge_mode == "positive_only":
                active_attrs = (targets[b] > 0.5).nonzero(as_tuple=False).view(-1)
                for a in active_attrs.tolist():
                    object_indices.append(flat_obj_idx)
                    attr_indices.append(b * num_attrs + a)
            else:
                raise ValueError(f"Unknown edge_mode={edge_mode}")

    if not object_indices:
        object_indices = [0]
        attr_indices = [0]

    edge_index = torch.tensor(
        [object_indices, attr_indices], dtype=torch.long, device=device
    )
    edge_weight = torch.ones(edge_index.shape[1], device=device)

    return BipartiteGraphBatch(
        object_feats=feats,
        attr_feats=attr_tensor,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

