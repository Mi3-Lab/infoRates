"""
VideoMamba3-AFA — Adaptive Frame Allocation with Mamba-3 SSM.

Two-stage architecture:
  Stage 1 (scanner):   small VideoMamba3 on T_sparse frames
                       → Temporal Concentration Head → per-frame scores
  Selector:            differentiable top-B frame selection
  Stage 2 (classifier): full VideoMamba3 on B selected frames → logits

Model variants
--------------
AFA-Tiny  : S1 depth=4  D=192 | S2 depth=12 D=192 | ~10M params
AFA-Small : S1 depth=4  D=192 | S2 depth=24 D=384 | ~26M params
AFA-Base  : S1 depth=8  D=384 | S2 depth=24 D=384 | ~85M params
"""
from __future__ import annotations

import math
import sys
import os
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

# resolve sibling imports
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from mamba3_core import BiMamba3
from videomamba3 import (
    VisionMamba,
    PatchEmbed,
    create_block,
    _init_weights,
    segm_init_weights,
)

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn, layer_norm_fn
except ImportError:
    RMSNorm = layer_norm_fn = rms_norm_fn = None

from afa_module import TemporalConcentrationHead, AdaptiveFrameSelector


# ---------------------------------------------------------------------------
# Stage-1 backbone: returns ALL hidden states (not just CLS)
# ---------------------------------------------------------------------------

class Stage1Scanner(nn.Module):
    """Small VideoMamba3 that returns full hidden-state sequence.

    Identical to VisionMamba except ``forward_features`` returns
    [B, 1 + T_sparse * N_patches, D] instead of just the CLS token.
    Used to feed the Temporal Concentration Head.

    Parameters
    ----------
    img_size, patch_size, embed_dim, depth, num_frames
        Standard VisionMamba3 parameters.
    mamba3_variant : str
        'trapezoidal' | 'complex' | 'mimo'  (default: 'complex')
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_frames: int = 12,
        mamba3_variant: Optional[str] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        ssm_cfg: Optional[dict] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        rms_norm = RMSNorm is not None
        fused_add_norm = RMSNorm is not None

        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=1,          # no tubelet — treat each frame independently
            in_chans=3,
            embed_dim=embed_dim,
        )
        n_patches = self.patch_embed.num_patches
        self.n_patches = n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # mamba3_variant=None → standard BiMamba (mamba_ssm CUDA kernels)
        # mamba3_variant="complex"|"trapezoidal"|"mimo" → BiMamba3 with vectorized GPU scan
        bimamba = mamba3_variant is None
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg or {},
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba=bimamba,
                mamba3_variant=mamba3_variant,
                mamba3_impl="reference",
                drop_path=dpr[i],
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        norm_cls = (nn.LayerNorm if not rms_norm else RMSNorm)
        self.norm_f = norm_cls(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(_init_weights, n_layer=depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, C, T_sparse, H, W]

        Returns
        -------
        hidden_states : Tensor [B, 1 + T_sparse * N_patches, D]
            Full token sequence including CLS.
        """
        B, C, T, H, W = x.shape
        # patch embed: [B, D, T, H', W'] → per-frame tokens
        x = self.patch_embed(x)                                   # [B, D, T, H', W']
        B_, D, T_, Hp, Wp = x.shape
        N = Hp * Wp   # patches per frame

        x = x.permute(0, 2, 3, 4, 1).reshape(B_ * T_, N, D)     # [B*T, N, D]

        # spatial pos + CLS
        cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tok, x], dim=1)                        # [B*T, N+1, D]
        x = x + self.pos_embed

        # temporal pos (applied to patch tokens only, not CLS)
        cls_tokens = x[:B_, :1, :]
        x = x[:, 1:]                                              # [B*T, N, D]
        x = rearrange(x, "(b t) n m -> (b n) t m", b=B_, t=T_)
        x = x + self.temporal_pos_embed[:, :T_]
        x = rearrange(x, "(b n) t m -> b (t n) m", b=B_, t=T_)
        x = torch.cat([cls_tokens, x], dim=1)                     # [B, 1+T*N, D]
        x = self.pos_drop(x)

        # SSM blocks
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # final norm — return ALL tokens (not just CLS)
        if RMSNorm is None or not isinstance(self.norm_f, RMSNorm):
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(self.norm_f.weight.dtype))
        else:
            hidden_states = rms_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True,
            )

        return hidden_states   # [B, 1 + T_sparse*N_patches, D]


# ---------------------------------------------------------------------------
# Stage-2 backbone: full VideoMamba3, receives selected frames
# ---------------------------------------------------------------------------

class Stage2Classifier(nn.Module):
    """Full VideoMamba3 operating on adaptively selected frames.

    Temporal positional embeddings are interpolated to the selected frame
    positions (relative to T_max) so the model knows where in the video
    each frame came from.

    Parameters
    ----------
    budget_B : int
        Number of selected frames (= sequence length for Stage 2).
    T_max : int
        Total frames in the video (used for positional embedding scaling).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 24,
        num_classes: int = 400,
        budget_B: int = 16,
        T_max: int = 48,
        mamba3_variant: Optional[str] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        fc_drop_rate: float = 0.0,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        ssm_cfg: Optional[dict] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        rms_norm = RMSNorm is not None
        fused_add_norm = RMSNorm is not None

        self.embed_dim = embed_dim
        self.budget_B = budget_B
        self.T_max = T_max
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=1,
            in_chans=3,
            embed_dim=embed_dim,
        )
        n_patches = self.patch_embed.num_patches
        self.n_patches = n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, T_max, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        bimamba = mamba3_variant is None
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg or {},
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba=bimamba,
                mamba3_variant=mamba3_variant,
                mamba3_impl="reference",
                drop_path=dpr[i],
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        norm_cls = (nn.LayerNorm if not rms_norm else RMSNorm)
        self.norm_f = norm_cls(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(_init_weights, n_layer=depth))

    def forward(
        self,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        frames : Tensor [B, budget_B, C, H, W]
            Adaptively selected frames.
        frame_indices : LongTensor [B, budget_B]
            Original positions of selected frames in [0, T_max).

        Returns
        -------
        logits : Tensor [B, num_classes]
        """
        B, T, C, H, W = frames.shape
        # patch embed each frame
        # reshape to [B*T, C, H, W] → Conv3d with kernel_size=1
        x = frames.reshape(B * T, C, H, W).unsqueeze(2)          # [B*T, C, 1, H, W]
        x = self.patch_embed(x)                                   # [B*T, D, 1, H', W']
        _, D, _, Hp, Wp = x.shape
        N = Hp * Wp
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, N, D)       # [B*T, N, D]

        # spatial pos + CLS
        cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x = x + self.pos_embed

        # temporal positional embedding — index into full T_max embedding
        # frame_indices: [B, T] → gather rows from temporal_pos_embed [1, T_max, D]
        cls_tokens = x[:B, :1, :]                                 # [B, 1, D]
        x = x[:, 1:]                                              # [B*T, N, D]
        x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)

        # gather per-selected-frame temporal pos embeddings
        # temporal_pos_embed: [1, T_max, D]
        t_pos = self.temporal_pos_embed.expand(B, -1, -1)         # [B, T_max, D]
        idx = frame_indices.unsqueeze(-1).expand(-1, -1, D)       # [B, T, D]
        t_pos_sel = t_pos.gather(dim=1, index=idx)                # [B, T, D]

        # broadcast across patches: [B*N, T, D] += [B, T, D] broadcast
        # reshape t_pos_sel to [B*N, T, D]
        t_pos_sel_exp = t_pos_sel.unsqueeze(1).expand(-1, N, -1, -1)   # [B, N, T, D]
        t_pos_sel_exp = t_pos_sel_exp.reshape(B * N, T, D)
        x = x + t_pos_sel_exp

        x = rearrange(x, "(b n) t m -> b (t n) m", b=B, t=T)
        x = torch.cat([cls_tokens, x], dim=1)                     # [B, 1+T*N, D]
        x = self.pos_drop(x)

        # SSM blocks
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # final norm + CLS head
        if RMSNorm is None or not isinstance(self.norm_f, RMSNorm):
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(self.norm_f.weight.dtype))
        else:
            hidden_states = rms_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight, self.norm_f.bias,
                eps=self.norm_f.eps, residual=residual,
                prenorm=False, residual_in_fp32=True,
            )

        cls_out = hidden_states[:, 0]                             # [B, D]
        return self.head(self.head_drop(cls_out))


# ---------------------------------------------------------------------------
# Full VideoMamba3-AFA model
# ---------------------------------------------------------------------------

class VideoMamba3AFA(nn.Module):
    """VideoMamba3 with Adaptive Frame Allocation.

    Parameters
    ----------
    img_size : int
    patch_size : int
    num_classes : int
    T_max : int
        Total frames decoded from the video (e.g. 48).
    T_sparse : int
        Frames processed by Stage-1 scanner (e.g. 12 = T_max // 4).
    budget_B : int
        Frames selected for Stage-2 classifier (e.g. 16).
    s1_depth, s1_dim : Stage-1 architecture.
    s2_depth, s2_dim : Stage-2 architecture.
    mamba3_variant : str or None
        None → standard BiMamba (mamba_ssm CUDA kernels).
        'complex'|'trapezoidal'|'mimo' → Mamba3 with vectorized GPU scan (paper default).
    selector_temperature : float
        Temperature for the soft straight-through in the selector.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 400,
        T_max: int = 48,
        T_sparse: int = 12,
        budget_B: int = 16,
        s1_depth: int = 4,
        s1_dim: int = 192,
        s2_depth: int = 24,
        s2_dim: int = 384,
        mamba3_variant: Optional[str] = "complex",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        fc_drop_rate: float = 0.0,
        selector_temperature: float = 1.0,
        ssm_cfg: Optional[dict] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.T_max    = T_max
        self.T_sparse = T_sparse
        self.budget_B = budget_B

        # number of spatial patches per frame (e.g. 196 for 224/16)
        n_patches = (img_size // patch_size) ** 2

        self.stage1 = Stage1Scanner(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=s1_dim,
            depth=s1_depth,
            num_frames=T_sparse,
            mamba3_variant=mamba3_variant,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ssm_cfg=ssm_cfg,
            device=device,
            dtype=dtype,
        )

        self.tch = TemporalConcentrationHead(
            d_model=s1_dim,
            T_sparse=T_sparse,
            T_max=T_max,
            n_patches_per_frame=n_patches,
            has_cls_token=True,
        )

        self.selector = AdaptiveFrameSelector(
            budget_B=budget_B,
            temperature=selector_temperature,
        )

        self.stage2 = Stage2Classifier(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=s2_dim,
            depth=s2_depth,
            num_classes=num_classes,
            budget_B=budget_B,
            T_max=T_max,
            mamba3_variant=mamba3_variant,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            fc_drop_rate=fc_drop_rate,
            ssm_cfg=ssm_cfg,
            device=device,
            dtype=dtype,
        )

    def sample_sparse_indices(self, T_max: int, T_sparse: int) -> list[int]:
        """Uniformly spaced indices for Stage-1 sparse scan."""
        return list(range(0, T_max, T_max // T_sparse))[:T_sparse]

    def forward(
        self,
        video: torch.Tensor,
        return_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        video : Tensor [B, C, T_max, H, W]
            Full decoded frames (T_max frames).
        return_scores : bool
            If True, also return TCH scores [B, T_max] (for analysis / distillation).

        Returns
        -------
        logits : Tensor [B, num_classes]
        scores : Tensor [B, T_max]   (only when return_scores=True)
        """
        B, C, T, H, W = video.shape
        assert T == self.T_max, f"Expected T_max={self.T_max} frames, got {T}"

        # ── Stage 1: sparse scan ────────────────────────────────────────────
        sparse_idx = self.sample_sparse_indices(self.T_max, self.T_sparse)
        sparse_frames = video[:, :, sparse_idx]                   # [B, C, T_sparse, H, W]
        s1_hidden = self.stage1(sparse_frames)                    # [B, 1+T_sparse*N, D1]

        # ── TCH: per-frame scores ───────────────────────────────────────────
        scores = self.tch(s1_hidden)                              # [B, T_max]

        # ── Selector: pick top-B frames ─────────────────────────────────────
        indices, soft_weights = self.selector(scores, training=self.training)

        # gather selected frames: [B, budget_B, C, H, W]
        video_t = video.permute(0, 2, 1, 3, 4)                   # [B, T, C, H, W]
        selected = self.selector.gather_frames(video_t, indices)  # [B, B_budget, C, H, W]

        # ── Stage 2: full classification ────────────────────────────────────
        logits = self.stage2(selected, indices)                   # [B, num_classes]

        if return_scores:
            return logits, scores
        return logits

    @property
    def n_params(self) -> dict[str, int]:
        s1 = sum(p.numel() for p in self.stage1.parameters())
        tch = sum(p.numel() for p in self.tch.parameters())
        s2 = sum(p.numel() for p in self.stage2.parameters())
        return {"stage1": s1, "tch": tch, "stage2": s2, "total": s1 + tch + s2}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def videomamba3_afa_tiny(num_classes: int = 400, **kwargs) -> VideoMamba3AFA:
    """~10M params. Fast inference, edge/mobile target."""
    return VideoMamba3AFA(
        num_classes=num_classes,
        T_max=48, T_sparse=12, budget_B=8,
        s1_depth=4,  s1_dim=192,
        s2_depth=12, s2_dim=192,
        mamba3_variant="complex",
        **kwargs,
    )


def videomamba3_afa_small(num_classes: int = 400, **kwargs) -> VideoMamba3AFA:
    """~26M params. Standard accuracy/efficiency tradeoff."""
    return VideoMamba3AFA(
        num_classes=num_classes,
        T_max=48, T_sparse=12, budget_B=16,
        s1_depth=4,  s1_dim=192,
        s2_depth=24, s2_dim=384,
        mamba3_variant="complex",
        **kwargs,
    )


def videomamba3_afa_base(num_classes: int = 400, **kwargs) -> VideoMamba3AFA:
    """~85M params. SOTA target."""
    return VideoMamba3AFA(
        num_classes=num_classes,
        T_max=64, T_sparse=16, budget_B=16,
        s1_depth=8,  s1_dim=384,
        s2_depth=24, s2_dim=384,
        mamba3_variant="complex",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        ("AFA-Tiny",  videomamba3_afa_tiny),
        ("AFA-Small", videomamba3_afa_small),
    ]

    for name, factory in configs:
        model = factory(num_classes=101).to(device)
        params = model.n_params
        print(f"\n{name}")
        print(f"  Stage1  : {params['stage1']/1e6:.1f}M")
        print(f"  TCH     : {params['tch']/1e6:.2f}M")
        print(f"  Stage2  : {params['stage2']/1e6:.1f}M")
        print(f"  Total   : {params['total']/1e6:.1f}M")

        T_max = model.T_max
        video = torch.randn(2, 3, T_max, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            logits, scores = model(video, return_scores=True)
        print(f"  logits  : {tuple(logits.shape)}")
        print(f"  scores  : {tuple(scores.shape)}  "
              f"min={scores.min():.3f} max={scores.max():.3f}")
        print(f"  budget  : {model.budget_B}/{T_max} frames "
              f"({100*model.budget_B/T_max:.0f}%)")
