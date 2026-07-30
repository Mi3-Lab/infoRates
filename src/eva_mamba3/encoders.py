"""Frame encoders: map (B, T, H, W, 3) uint8 → (B, T, C) float.

Each encoder exposes a C_out attribute so EVAWrapper can wire dimensions
automatically. New backbones only need to implement this interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch import Tensor

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ── ResNet-50 (CNN, always available) ─────────────────────────────────────────

class ResNetFrameEncoder(nn.Module):
    """ResNet-50 applied independently to each frame.

    (B, T, H, W, 3) uint8  →  (B, T, C_out) float32
    """

    CHUNK = 16

    def __init__(self, C_out: int = 512, freeze: bool = True):
        super().__init__()
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(2048, C_out), nn.LayerNorm(C_out))
        self.register_buffer("mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std",  _IMAGENET_STD.view(1, 3, 1, 1))
        self.C_out = C_out
        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.body.parameters():
            p.requires_grad_(True)

    def _encode_chunk(self, chunk: Tensor) -> Tensor:
        x = (chunk - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            x = self.body(x)
        return self.proj(self.pool(x).flatten(1))

    def forward(self, frames: Tensor) -> Tensor:
        B, T, H, W, _ = frames.shape
        x = frames.view(B * T, H, W, 3).permute(0, 3, 1, 2).float().div(255.0)
        feats = torch.cat([self._encode_chunk(c) for c in x.split(self.CHUNK)], dim=0)
        return feats.view(B, T, self.C_out)


# ── ViT (timm, any variant) ───────────────────────────────────────────────────

class ViTFrameEncoder(nn.Module):
    """Any timm ViT applied per-frame; outputs the CLS token.

    Requires:  pip install timm

    Example model_names:
      "vit_base_patch16_224"          — ViT-B/16 ImageNet-21k
      "vit_large_patch16_224"         — ViT-L/16
      "dino_vitb16" (via timm alias)  — DINO ViT-B/16
    """

    CHUNK = 8

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("ViTFrameEncoder requires timm: pip install timm")

        vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit   = vit
        self.C_out: int = vit.embed_dim
        self.register_buffer("mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std",  _IMAGENET_STD.view(1, 3, 1, 1))

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.vit.parameters():
            p.requires_grad_(True)

    def _encode_chunk(self, chunk: Tensor) -> Tensor:
        x = (chunk.float().div(255.0) - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            return self.vit(x)   # (N, C_out)  — CLS token

    def forward(self, frames: Tensor) -> Tensor:
        B, T, H, W, _ = frames.shape
        x = frames.view(B * T, H, W, 3).permute(0, 3, 1, 2)
        feats = torch.cat([self._encode_chunk(c) for c in x.split(self.CHUNK)], dim=0)
        return feats.view(B, T, self.C_out)


# ── VideoMAE / HuggingFace ViT-MAE ───────────────────────────────────────────

class VideoMAEFrameEncoder(nn.Module):
    """HuggingFace VideoMAE spatial encoder; mean-pools patch tokens per frame.

    Requires:  pip install transformers

    model_name examples:
      "MCG-NJU/videomae-base"
      "MCG-NJU/videomae-large"
    """

    CHUNK = 4

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError:
            raise ImportError("VideoMAEFrameEncoder requires transformers: pip install transformers")

        self.model  = VideoMAEModel.from_pretrained(model_name)
        self.C_out  = self.model.config.hidden_size
        self.register_buffer("mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std",  _IMAGENET_STD.view(1, 3, 1, 1))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    def _encode_one_frame(self, frame: Tensor) -> Tensor:
        """frame: (B, 3, H, W) uint8 → (B, C_out)"""
        x = (frame.float().div(255.0) - self.mean) / self.std
        # VideoMAE expects (B, C, T, H, W); wrap single frame as T=1 video
        x = x.unsqueeze(2)   # (B, 3, 1, H, W)
        # bool_masked_pos: (B, num_patches) — all False (no masking at inference)
        B = x.shape[0]
        n_patches = self.model.config.num_frames * (
            (self.model.config.image_size // self.model.config.patch_size) ** 2
            // self.model.config.tubelet_size
        )
        mask = torch.zeros(B, n_patches, dtype=torch.bool, device=x.device)
        out  = self.model(pixel_values=x, bool_masked_pos=mask)
        return out.last_hidden_state[:, 1:].mean(1)   # mean pool patch tokens, skip CLS

    def forward(self, frames: Tensor) -> Tensor:
        B, T, H, W, _ = frames.shape
        per_frame = frames.permute(0, 1, 4, 2, 3)   # (B, T, 3, H, W)
        feats = []
        for t in range(T):
            feats.append(self._encode_one_frame(per_frame[:, t]))   # (B, C_out)
        return torch.stack(feats, dim=1)   # (B, T, C_out)


# ── TimeSformer (HuggingFace, pretrained video encoder) ──────────────────────

class TimeSformerFrameEncoder(nn.Module):
    """TimeSformer video encoder converted into per-frame features.

    The pretrained TimeSformer processes short clips (8 frames by default). For
    dense EVA clips, we run non-overlapping chunks and mean-pool spatial patch
    tokens per frame, yielding one feature vector per input frame.

    Input:  (B, T, H, W, 3) uint8 RGB
    Output: (B, T, C_out)
    """

    def __init__(
        self,
        model_name: str = "facebook/timesformer-base-finetuned-k400",
        freeze: bool = True,
        chunk_frames: int | None = None,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("TimeSformerFrameEncoder requires transformers") from exc

        self.model = AutoModel.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        self.C_out = int(cfg.hidden_size)
        self.chunk_frames = int(chunk_frames or cfg.num_frames)
        self.patch_size = int(cfg.patch_size)

        self.register_buffer("mean", torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    def _encode_chunk(self, chunk: Tensor) -> Tensor:
        B, L, H, W, _ = chunk.shape
        if L < self.chunk_frames:
            pad = chunk[:, -1:].expand(B, self.chunk_frames - L, H, W, 3)
            chunk = torch.cat([chunk, pad], dim=1)
        x = chunk[:, : self.chunk_frames].permute(0, 1, 4, 2, 3).float().div(255.0)
        x = (x - self.mean) / self.std

        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:]  # drop CLS, (B, L*P, C)
        patches_per_frame = tokens.shape[1] // self.chunk_frames
        tokens = tokens.view(B, self.chunk_frames, patches_per_frame, self.C_out)
        frame_features = tokens.mean(dim=2)
        return frame_features[:, :L]

    def forward(self, frames: Tensor) -> Tensor:
        B, T = frames.shape[:2]
        chunks = []
        for start in range(0, T, self.chunk_frames):
            chunks.append(self._encode_chunk(frames[:, start : start + self.chunk_frames]))
        return torch.cat(chunks, dim=1)[:, :T]
