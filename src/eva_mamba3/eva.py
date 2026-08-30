"""EVA: Evidence-Preserving Video Anti-aliasing tokenizer.

Takes dense per-frame features (B, T, C) and stride s,
produces anti-aliased tokens (B, T//s [* M], D).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

STRIDE_TO_IDX = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}


class LearnableGaussian(nn.Module):
    """1-D depthwise Gaussian low-pass filter with learnable bandwidth σ."""

    def __init__(self, K: int = 3, sigma_init: float = 1.5):
        super().__init__()
        self.K = K
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))

    def forward(self, f: Tensor) -> Tensor:
        """f: (B, T, C) → ℓ: (B, T, C)  (low-frequency context)"""
        B, T, C = f.shape
        sigma = self.log_sigma.exp().clamp(0.3, 5.0)
        k = torch.arange(-self.K, self.K + 1, device=f.device, dtype=f.dtype)
        h = torch.exp(-k ** 2 / (2 * sigma ** 2))
        h = h / h.sum()                            # (2K+1,)

        # Apply as depthwise 1-D conv over time: (B, C, T)
        ft = f.permute(0, 2, 1)                    # (B, C, T)
        ft_pad = F.pad(ft, (self.K, self.K), mode="replicate")
        # weight: (C, 1, 2K+1)
        weight = h.view(1, 1, -1).expand(C, 1, -1)
        ell = F.conv1d(ft_pad, weight, groups=C)   # (B, C, T)
        return ell.permute(0, 2, 1)                # (B, T, C)


class EventScorer(nn.Module):
    """MLP that assigns a discriminative salience score to each frame."""

    def __init__(self, C: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * C, C // 4),
            nn.GELU(),
            nn.Linear(C // 4, 1),
        )

    def forward(self, f: Tensor, r: Tensor) -> Tensor:
        """
        f: (B, T, C)  absolute features
        r: (B, T, C)  high-frequency residuals
        returns: (B, T)  salience scores in [0, 1]
        """
        # Forward derivative: difference to next frame (pad last)
        df_fwd = torch.zeros_like(f)
        df_fwd[:, :-1] = (f[:, 1:] - f[:, :-1]).abs()
        df_fwd[:, -1]  = df_fwd[:, -2]

        # Backward derivative: difference to previous frame (pad first)
        df_bwd = torch.zeros_like(f)
        df_bwd[:, 1:]  = (f[:, :-1] - f[:, 1:]).abs()
        df_bwd[:, 0]   = df_bwd[:, 1]

        inp = torch.cat([f, df_fwd, df_bwd, r.abs()], dim=-1)  # (B, T, 4C)
        return torch.sigmoid(self.mlp(inp)).squeeze(-1)          # (B, T)


class EVATokenizer(nn.Module):
    """Dual-band evidence-preserving tokenizer.

    For each stride window of width s, produces either:
      M=1: one fused token  z = LN(W_ctx·z_ctx + W_evt·z_evt + p)
      M=2: two tokens        [z_ctx_proj, z_evt_proj] interleaved
    """

    def __init__(
        self,
        C: int,
        D: int,
        K: int = 3,
        sigma_init: float = 1.5,
        tau: float = 0.1,
    ):
        super().__init__()
        self.C   = C
        self.D   = D
        self.tau = tau

        self.gaussian = LearnableGaussian(K, sigma_init)
        self.scorer   = EventScorer(C)

        self.q_ctx = nn.Parameter(torch.randn(C) / C ** 0.5)
        self.W_ctx = nn.Linear(C, D, bias=False)
        self.W_evt = nn.Linear(C, D, bias=False)

        # Stride-conditioned positional embedding: 5 strides × D dims
        self.pos_embed = nn.Embedding(len(STRIDE_TO_IDX), D)
        self.ln        = nn.LayerNorm(D)

        # Cached for loss computation (set during forward)
        self._a_evt: Tensor | None = None

    def forward(self, f: Tensor, s: int, M: int = 1) -> Tensor:
        """
        f: (B, T, C)
        s: stride ∈ {1, 2, 4, 8, 16}
        M: tokens per window (1 or 2)
        returns: (B, T//s * M, D)
        """
        B, T, C = f.shape
        assert T % s == 0, f"T={T} not divisible by s={s}"
        n_win = T // s

        # Dual-band decomposition
        ell = self.gaussian(f)   # (B, T, C)  low-freq context
        r   = f - ell            # (B, T, C)  high-freq event residual

        # Event scores
        e = self.scorer(f, r)    # (B, T)

        # Reshape into windows: (B, n_win, s, C) and (B, n_win, s)
        ell_w = ell.view(B, n_win, s, C)
        r_w   = r.view(B, n_win, s, C)
        e_w   = e.view(B, n_win, s)

        # Context attention: softmax( q^T ℓ_t / √C )
        a_ctx = torch.einsum("c,bwsc->bws", self.q_ctx, ell_w) / C ** 0.5
        a_ctx = a_ctx.softmax(dim=-1)                            # (B, n_win, s)

        # Event attention: softmax( e_t / τ )   (sharpened)
        a_evt = (e_w / self.tau).softmax(dim=-1)                 # (B, n_win, s)
        self._a_evt = a_evt                                      # cache for loss

        # Aggregation
        z_ctx = torch.einsum("bws,bwsc->bwc", a_ctx, ell_w)     # (B, n_win, C)
        z_evt = torch.einsum("bws,bwsc->bwc", a_evt, r_w)       # (B, n_win, C)

        # Stride-conditioned positional embedding
        p_idx = torch.tensor(STRIDE_TO_IDX[s], device=f.device)
        p     = self.pos_embed(p_idx)                            # (D,)

        if M == 1:
            z = self.ln(self.W_ctx(z_ctx) + self.W_evt(z_evt) + p)  # (B, n_win, D)
            return z

        # M == 2: ctx and evt as separate tokens, interleaved
        z_c = self.W_ctx(z_ctx) + p   # (B, n_win, D)
        z_e = self.W_evt(z_evt) + p   # (B, n_win, D)
        # Interleave along sequence: [ctx_0, evt_0, ctx_1, evt_1, ...]
        z = torch.stack([z_c, z_e], dim=2).view(B, 2 * n_win, self.D)
        return self.ln(z)
