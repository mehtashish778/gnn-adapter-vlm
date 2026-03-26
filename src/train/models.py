from __future__ import annotations

from typing import List

import torch
from PIL import Image

from .encoders import ImageBackbone


class BaselineClassifier(torch.nn.Module):
    def __init__(
        self,
        clip_backbone: ImageBackbone | None,
        dino_backbone: ImageBackbone | None,
        feature_dim: int,
        hidden_dims: List[int],
        dropout: float,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.clip_backbone = clip_backbone
        self.dino_backbone = dino_backbone
        layers: List[torch.nn.Module] = []
        input_dim = feature_dim
        for hidden in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            input_dim = hidden
        layers.append(torch.nn.Linear(input_dim, num_labels))
        self.classifier = torch.nn.Sequential(*layers)

    def encode(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        features = []
        if self.clip_backbone is not None:
            features.append(self.clip_backbone.forward_pil(images, device=device))
        if self.dino_backbone is not None:
            features.append(self.dino_backbone.forward_pil(images, device=device))
        if len(features) == 1:
            return features[0]
        return torch.cat(features, dim=1)

    def forward(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        feats = self.encode(images=images, device=device)
        return self.classifier(feats)

    def freeze_backbones(self) -> None:
        for backbone in (self.clip_backbone, self.dino_backbone):
            if backbone is None:
                continue
            for param in backbone.parameters():
                param.requires_grad = False

