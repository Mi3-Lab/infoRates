"""Spatial encoder: ResNet-50 → per-frame features (B, T, C)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch import Tensor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


class SpatialEncoder(nn.Module):
    """Shared ResNet-50 applied independently to each frame.

    Input:  frames (B, T, H, W, 3)  uint8
    Output: features (B, T, C_out)   float32
    """

    CHUNK = 16  # frames processed simultaneously to avoid OOM

    def __init__(self, C_out: int = 512, freeze: bool = True):
        super().__init__()
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool and FC; keep up through layer4
        self.body = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(2048, C_out),
            nn.LayerNorm(C_out),
        )
        self.register_buffer("mean", IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std",  IMAGENET_STD.view(1, 3, 1, 1))

        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.body.parameters():
            p.requires_grad_(True)

    def _encode_chunk(self, chunk: Tensor) -> Tensor:
        """chunk: (N, 3, H, W) float [0,1] → (N, C_out)"""
        x = (chunk - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            x = self.body(x)               # (N, 2048, h, w)
        x = self.pool(x).flatten(1)        # (N, 2048)
        return self.proj(x)                # (N, C_out)

    def forward(self, frames: Tensor) -> Tensor:
        """
        frames: (B, T, H, W, 3) uint8
        returns: (B, T, C_out) float32
        """
        B, T, H, W, _ = frames.shape
        # Convert uint8 → float [0, 1], layout (B*T, 3, H, W)
        x = frames.view(B * T, H, W, 3).permute(0, 3, 1, 2).float().div(255.0)

        # Process in chunks to avoid OOM
        chunks = x.split(self.CHUNK, dim=0)
        feats  = torch.cat([self._encode_chunk(c) for c in chunks], dim=0)
        return feats.view(B, T, -1)        # (B, T, C_out)
