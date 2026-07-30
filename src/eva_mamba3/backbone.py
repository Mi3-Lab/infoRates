"""Temporal backbones for EVAWrapper.

  Mamba3Backbone           — bidirectional Mamba-3 + attention pooling (primary)
  TemporalTransformerBackbone — lightweight transformer, no external deps (ablation / ViT pairing)

Both accept (B, L, D_in) and return (B, n_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from mamba3_ssm import Mamba3


class BiMamba3Block(nn.Module):
    """One bidirectional Mamba-3 layer with pre-norm and residual."""

    def __init__(self, D: int, d_state: int = 64, expand: int = 2):
        super().__init__()
        # d_state=64 complex states ≡ 128 real-equivalent (Mamba-3 halves state dim)
        mamba_kwargs = dict(d_model=D, d_state=d_state, expand=expand)
        self.fwd  = Mamba3(**mamba_kwargs)
        self.bwd  = Mamba3(**mamba_kwargs)
        self.norm_fwd = nn.LayerNorm(D)
        self.norm_bwd = nn.LayerNorm(D)
        self.merge    = nn.Linear(2 * D, D, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, D) → (B, L, D)"""
        h_fwd = self.fwd(self.norm_fwd(x)) + x                    # (B, L, D)
        h_bwd = self.bwd(self.norm_bwd(x).flip(1)).flip(1) + x    # (B, L, D)
        return self.merge(torch.cat([h_fwd, h_bwd], dim=-1)) + x  # (B, L, D)


class Mamba3Backbone(nn.Module):
    """Stack of bidirectional Mamba-3 blocks + attention pooling + linear head.

    Input:  Z  (B, L, D_in)
    Output: logits (B, n_classes)
    """

    def __init__(
        self,
        D_in: int,
        D: int = 384,
        n_layers: int = 12,
        d_state: int = 64,
        expand: int = 2,
        n_classes: int = 97,
    ):
        super().__init__()
        self.input_proj = nn.Linear(D_in, D)
        self.blocks = nn.ModuleList(
            [BiMamba3Block(D, d_state=d_state, expand=expand) for _ in range(n_layers)]
        )
        D_out = D  # merge projection keeps dimension D throughout
        # Attention pooling: learns which tokens are most relevant
        self.attn_pool = nn.Linear(D_out, 1)
        self.head_norm  = nn.LayerNorm(D_out)
        self.head       = nn.Linear(D_out, n_classes)

    def forward(self, Z: Tensor) -> Tensor:
        """Z: (B, L, D_in) → logits (B, n_classes)"""
        x = self.input_proj(Z)               # (B, L, D)
        for blk in self.blocks:
            x = blk(x)                       # (B, L, D)
        # Attention pooling over sequence
        w = self.attn_pool(x).softmax(dim=1) # (B, L, 1)
        v = (w * x).sum(dim=1)               # (B, D)
        return self.head(self.head_norm(v))  # (B, n_classes)


# ── Temporal Transformer (no Mamba dependency) ────────────────────────────────

class TemporalTransformerBackbone(nn.Module):
    """Lightweight temporal transformer backbone.

    Uses a learned CLS token + standard pre-norm transformer encoder.
    No external dependencies — pairs with any frame encoder for ablation
    or when mamba3-ssm is not available.

    Input:  Z  (B, L, D_in)
    Output: logits (B, n_classes)
    """

    def __init__(
        self,
        D_in: int,
        D: int = 384,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        n_classes: int = 97,
    ):
        super().__init__()
        self.input_proj = nn.Linear(D_in, D)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, D) / D ** 0.5)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=n_heads,
            dim_feedforward=int(D * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head_norm = nn.LayerNorm(D)
        self.head      = nn.Linear(D, n_classes)

    def forward(self, Z: Tensor) -> Tensor:
        """Z: (B, L, D_in) → logits (B, n_classes)"""
        x   = self.input_proj(Z)                                    # (B, L, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)            # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                           # (B, 1+L, D)
        x   = self.encoder(x)                                       # (B, 1+L, D)
        return self.head(self.head_norm(x[:, 0]))                   # (B, n_classes)
