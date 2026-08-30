"""Baselines for fair EVA comparisons."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class UniformStrideWrapper(nn.Module):
    """Uniform-stride baseline with the same encoder/backbone interface as EVA.

    The model encodes dense frames, keeps one frame every ``s`` positions, projects
    frame features to the temporal backbone dimension, and classifies the sparse
    sequence. This is the direct "frame dropping" baseline for EVA.
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        temporal_backbone: nn.Module,
        C_in: int,
        D: int,
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.input_proj = nn.Linear(C_in, D)
        self.temporal_backbone = temporal_backbone

    def unfreeze_encoder(self) -> None:
        if hasattr(self.frame_encoder, "unfreeze"):
            self.frame_encoder.unfreeze()
        else:
            for p in self.frame_encoder.parameters():
                p.requires_grad_(True)

    def forward(self, frames: Tensor, s: int) -> dict[str, Tensor | None]:
        f = self.frame_encoder(frames)
        z = self.input_proj(f[:, ::s])
        logits = self.temporal_backbone(z)
        return {
            "logits": logits,
            "logits_dense": None,
            "a_evt": None,
            "a_evt_dense": None,
            "evidence": None,
        }


class TimeSformerClassifierWrapper(nn.Module):
    """Pure pretrained TimeSformer baseline with stride-controlled sampling."""

    def __init__(
        self,
        n_classes: int,
        model_name: str = "facebook/timesformer-base-finetuned-k400",
        num_frames: int = 8,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForVideoClassification
        except ImportError as exc:
            raise ImportError("TimeSformerClassifierWrapper requires transformers") from exc

        self.num_frames = num_frames
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )
        self.register_buffer("mean", torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1))

    def _sample_stride(self, frames: Tensor, s: int) -> Tensor:
        B, T, H, W, C = frames.shape
        candidates = torch.arange(0, T, s, device=frames.device)
        if len(candidates) >= self.num_frames:
            pick = torch.linspace(
                0,
                len(candidates) - 1,
                self.num_frames,
                device=frames.device,
            ).round().long()
            idx = candidates.index_select(0, pick)
        else:
            pad = candidates[-1:].expand(self.num_frames - len(candidates))
            idx = torch.cat([candidates, pad], dim=0)
        return frames.index_select(1, idx)

    def forward(self, frames: Tensor, s: int) -> dict[str, Tensor | None]:
        clip = self._sample_stride(frames, s)
        x = clip.permute(0, 1, 4, 2, 3).float().div(255.0)
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)
        return {
            "logits": out.logits,
            "logits_dense": None,
            "a_evt": None,
            "a_evt_dense": None,
            "evidence": None,
        }
