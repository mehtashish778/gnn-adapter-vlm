from __future__ import annotations

from typing import List, Tuple

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config


class ImageBackbone(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = create_transform(**cfg, is_training=False)
        self.output_dim = getattr(self.model, "num_features", None)
        if self.output_dim is None:
            raise RuntimeError(f"Could not infer output dim for {model_name}")

    def forward_pil(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in images], dim=0).to(device)
        return self.model(batch)


def build_backbones(
    use_clip: bool,
    use_dino: bool,
    clip_model_name: str,
    dino_model_name: str,
    clip_pretrained: bool,
    dino_pretrained: bool,
) -> Tuple[ImageBackbone | None, ImageBackbone | None, int]:
    clip = None
    dino = None
    feature_dim = 0
    if use_clip:
        clip = ImageBackbone(model_name=clip_model_name, pretrained=clip_pretrained)
        feature_dim += int(clip.output_dim)
    if use_dino:
        dino = ImageBackbone(model_name=dino_model_name, pretrained=dino_pretrained)
        feature_dim += int(dino.output_dim)
    if feature_dim <= 0:
        raise RuntimeError("At least one encoder must be enabled for baseline training.")
    return clip, dino, feature_dim

