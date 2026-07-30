"""EVAWrapper: architecture-agnostic composition of frame encoder + EVA + temporal backbone.

Any combination works as long as the interfaces match:
  frame_encoder    : (B, T, H, W, 3) uint8   →  (B, T, C)        float
  eva              : (B, T, C)                →  (B, T//s [*M], D) float
  temporal_backbone: (B, L, D)               →  (B, n_classes)    float
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .eva import EVATokenizer


class EVAWrapper(nn.Module):
    """Compose any frame encoder, EVA tokenizer, and temporal backbone.

    Handles:
      - Dense forward pass (s=1) for KL distillation during training
      - M=1/M=2 token mode selection based on stride
      - Encoder freeze/unfreeze lifecycle
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        eva: EVATokenizer,
        temporal_backbone: nn.Module,
    ):
        super().__init__()
        self.frame_encoder    = frame_encoder
        self.eva              = eva
        self.temporal_backbone = temporal_backbone

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run(self, f: Tensor, s: int) -> tuple[Tensor, Tensor]:
        """EVA + backbone for one stride. Returns (logits, a_evt)."""
        M = 2 if s >= 8 else 1
        Z     = self.eva(f, s=s, M=M)    # (B, T//s * M, D)
        a_evt = self.eva._a_evt          # (B, n_win, s) cached by EVATokenizer
        return self.temporal_backbone(Z), a_evt

    # ── Public API ────────────────────────────────────────────────────────────

    def unfreeze_encoder(self) -> None:
        """Unfreeze the frame encoder (call after warm-up epochs)."""
        if hasattr(self.frame_encoder, "unfreeze"):
            self.frame_encoder.unfreeze()
        else:
            for p in self.frame_encoder.parameters():
                p.requires_grad_(True)

    def forward(self, frames: Tensor, s: int) -> dict[str, Tensor | None]:
        """
        Args:
            frames: (B, T, H, W, 3)  uint8
            s:      stride  ∈ {1, 2, 4, 8, 16}

        Returns dict with keys:
            logits        (B, n_classes)   — prediction at stride s
            logits_dense  (B, n_classes)   — prediction at s=1  (training only, s>1)
            a_evt         (B, n_win, s)    — event attention weights, stride s
            a_evt_dense   (B, T, 1)        — event attention weights, s=1 (training only)
        """
        f = self.frame_encoder(frames)   # (B, T, C)

        logits_dense = a_evt_dense = None
        if self.training and s > 1:
            logits_dense, a_evt_dense = self._run(f, s=1)

        logits, a_evt = self._run(f, s=s)

        return {
            "logits":       logits,
            "logits_dense": logits_dense,
            "a_evt":        a_evt,
            "a_evt_dense":  a_evt_dense,
        }
