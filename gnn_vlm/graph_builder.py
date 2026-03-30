from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .graph_models import BipartiteGraphBatch


def build_bipartite_batch(
    feats: torch.Tensor,
    targets: torch.Tensor,
    attr_feats: Optional[torch.Tensor] = None,
    edge_mode: str = "positive_only",
    ontology: Optional[Dict[str, Any]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
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
    edge_weights: List[float] = []

    ontology_per_attr: Optional[torch.Tensor] = None
    if edge_mode == "ontology_weighted":
        if ontology is None:
            raise ValueError("ontology_weighted edge mode requires ontology payload.")
        sem = ontology.get("semantic_sim_matrix")
        pmi = ontology.get("pmi_matrix")
        if sem is None or pmi is None:
            raise ValueError("Ontology payload must include semantic_sim_matrix and pmi_matrix.")
        sem_t = torch.as_tensor(sem, dtype=feats.dtype, device=device)
        pmi_t = torch.as_tensor(pmi, dtype=feats.dtype, device=device)
        if sem_t.shape != (num_attrs, num_attrs) or pmi_t.shape != (num_attrs, num_attrs):
            raise ValueError(
                f"Ontology matrices must be ({num_attrs}, {num_attrs}), got {sem_t.shape} and {pmi_t.shape}"
            )
        # Collapse pairwise ontology to a stable per-attribute prior.
        per_attr_sem = sem_t.mean(dim=1)
        per_attr_pmi = pmi_t.mean(dim=1)
        ontology_per_attr = (alpha * per_attr_sem) + (beta * per_attr_pmi)
        ontology_per_attr = ontology_per_attr - ontology_per_attr.min()
        ontology_per_attr = ontology_per_attr / ontology_per_attr.max().clamp_min(1e-6)
        ontology_per_attr = ontology_per_attr.clamp_min(1e-3)

    for b in range(bsz):
        for o in range(num_objects):
            flat_obj_idx = b * num_objects + o
            if edge_mode == "full_bipartite":
                for a in range(num_attrs):
                    object_indices.append(flat_obj_idx)
                    attr_indices.append(b * num_attrs + a)
                    edge_weights.append(1.0)
            elif edge_mode == "positive_only":
                active_attrs = (targets[b] > 0.5).nonzero(as_tuple=False).view(-1)
                for a in active_attrs.tolist():
                    object_indices.append(flat_obj_idx)
                    attr_indices.append(b * num_attrs + a)
                    edge_weights.append(1.0)
            elif edge_mode == "ontology_weighted":
                assert ontology_per_attr is not None
                for a in range(num_attrs):
                    object_indices.append(flat_obj_idx)
                    attr_indices.append(b * num_attrs + a)
                    edge_weights.append(float(ontology_per_attr[a].item()))
            else:
                raise ValueError(f"Unknown edge_mode={edge_mode}")

    if not object_indices:
        object_indices = [0]
        attr_indices = [0]
        edge_weights = [1.0]

    edge_index = torch.tensor(
        [object_indices, attr_indices], dtype=torch.long, device=device
    )
    edge_weight = torch.tensor(edge_weights, dtype=feats.dtype, device=device)

    return BipartiteGraphBatch(
        object_feats=feats,
        attr_feats=attr_tensor,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

