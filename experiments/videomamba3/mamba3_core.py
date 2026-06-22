"""Mamba-3 core SSM for VideoMamba3.

This is a pure-PyTorch reference implementation shaped after the released
Mamba-3 block in state-spaces/mamba. It keeps the official projections and
state structure (Q=C, K=B, V=x, dt/A/trap/angles, BC normalization, B/C bias,
and optional MIMO rank), while avoiding custom CUDA kernels so it can run in
the current VideoMamba training stack.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.modules.mamba3 import Mamba3 as OfficialMamba3
    _OFFICIAL_MAMBA3_ERROR = None
except Exception as exc:  # pragma: no cover - depends on CUDA/runtime install
    OfficialMamba3 = None
    _OFFICIAL_MAMBA3_ERROR = exc


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class Mamba3Core(nn.Module):
    """Reference Mamba-3 SSM mixer.

    Args mirror the official Mamba3 module where practical. The recurrent scan
    is intentionally explicit and slow; correctness/faithfulness is prioritized
    over kernel-level speed here.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        use_rope: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.headdim = min(headdim, self.d_inner)
        while self.d_inner % self.headdim != 0 and self.headdim > 1:
            self.headdim //= 2
        self.nheads = self.d_inner // self.headdim
        self.ngroups = ngroups
        self.A_floor = A_floor
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.use_rope = use_rope

        if rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0")
        self.rotary_dim = int(d_state * rope_fraction)
        if self.rotary_dim % 2:
            self.rotary_dim -= 1
        self.num_rope_angles = max(1, self.rotary_dim // 2)

        d_bc = self.d_state * self.ngroups * self.mimo_rank
        d_in_proj = 2 * self.d_inner + 2 * d_bc + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(_inv_softplus(dt), requires_grad=True)
        self.dt_bias._no_weight_decay = True

        bias_shape = (self.nheads, self.mimo_rank, self.d_state)
        self.B_bias = nn.Parameter(torch.ones(bias_shape, device=device, dtype=torch.float32))
        self.C_bias = nn.Parameter(torch.ones(bias_shape, device=device, dtype=torch.float32))
        self.B_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            )
            self.mimo_z = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device))
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            )

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def _rotate_qk(self, q: torch.Tensor, k: torch.Tensor, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mamba-3's data-dependent RoPE to Q/K state channels."""
        if not self.use_rope or self.rotary_dim < 2:
            return q, k
        rot = self.rotary_dim
        q_rot, q_pass = q[..., :rot], q[..., rot:]
        k_rot, k_pass = k[..., :rot], k[..., rot:]
        q1, q2 = q_rot[..., 0::2], q_rot[..., 1::2]
        k1, k2 = k_rot[..., 0::2], k_rot[..., 1::2]
        cos, sin = angles.cos(), angles.sin()
        q_rot = torch.stack((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1).flatten(-2)
        k_rot = torch.stack((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1).flatten(-2)
        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)

    def _project(self, u: torch.Tensor):
        zxbcdta = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxbcdta,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.ngroups * self.mimo_rank,
                self.d_state * self.ngroups * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.ngroups)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.ngroups)
        B = self.B_norm(B).expand(-1, -1, -1, self.nheads, -1)
        C = self.C_norm(C).expand(-1, -1, -1, self.nheads, -1)
        dt = F.softplus(dd_dt + self.dt_bias)
        A = torch.clamp(-F.softplus(dd_A.float()), max=-self.A_floor)
        trap = torch.sigmoid(trap)
        angles = angles.unsqueeze(2).expand(-1, -1, self.nheads, -1).float()
        return z, x, B, C, dt, A, trap, angles

    def forward(self, u: torch.Tensor, inference_params=None) -> torch.Tensor:
        z, x, B, C, dt, A, trap, angles = self._project(u)
        batch, seqlen = u.shape[:2]
        decay = torch.exp(A * dt)

        if self.is_mimo:
            y = self._scan_mimo(batch, seqlen, z, x, B, C, dt, decay, trap, angles)
        else:
            y = self._scan_siso(batch, seqlen, z, x, B, C, dt, decay, trap, angles)
        return self.out_proj(y.to(x.dtype))

    def _scan_siso(self, batch, seqlen, z, x, B, C, dt, decay, trap, angles):
        state = x.new_zeros(batch, self.nheads, self.headdim, self.d_state)
        prev_k = x.new_zeros(batch, self.nheads, self.d_state)
        prev_v = x.new_zeros(batch, self.nheads, self.headdim)
        angle_state = angles.new_zeros(batch, self.nheads, self.num_rope_angles)
        ys = []

        for t in range(seqlen):
            k_t = B[:, t, 0] + self.B_bias[:, 0].unsqueeze(0)
            q_t = C[:, t, 0] + self.C_bias[:, 0].unsqueeze(0)
            angle_state = angle_state + dt[:, t].unsqueeze(-1) * angles[:, t]
            q_t, k_t = self._rotate_qk(q_t, k_t, angle_state)

            d_t = decay[:, t].unsqueeze(-1).unsqueeze(-1)
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            trap_t = trap[:, t].unsqueeze(-1).unsqueeze(-1)
            now = torch.einsum("bhp,bhn->bhpn", x[:, t], k_t)
            prev = torch.einsum("bhp,bhn->bhpn", prev_v, prev_k)
            state = d_t * state + dt_t * ((1.0 - trap_t) * now + trap_t * d_t * prev)

            y_t = torch.einsum("bhpn,bhn->bhp", state, q_t)
            y_t = y_t + self.D.view(1, self.nheads, 1) * x[:, t]
            y_t = y_t * F.silu(z[:, t])
            ys.append(rearrange(y_t, "b h p -> b (h p)"))
            prev_k, prev_v = k_t, x[:, t]
        return torch.stack(ys, dim=1)

    def _scan_mimo(self, batch, seqlen, z, x, B, C, dt, decay, trap, angles):
        state = x.new_zeros(batch, self.mimo_rank, self.nheads, self.headdim, self.d_state)
        prev_k = x.new_zeros(batch, self.mimo_rank, self.nheads, self.d_state)
        prev_v = x.new_zeros(batch, self.mimo_rank, self.nheads, self.headdim)
        angle_state = angles.new_zeros(batch, self.nheads, self.num_rope_angles)
        ys = []

        x_r = torch.einsum("blhp,hrp->blrhp", x, self.mimo_x.to(x.dtype))
        z_r = torch.einsum("blhp,hrp->blrhp", z, self.mimo_z.to(z.dtype))

        for t in range(seqlen):
            k_t = B[:, t] + self.B_bias.permute(1, 0, 2).unsqueeze(0)
            q_t = C[:, t] + self.C_bias.permute(1, 0, 2).unsqueeze(0)
            angle_state = angle_state + dt[:, t].unsqueeze(-1) * angles[:, t]
            q_t, k_t = self._rotate_qk(q_t, k_t, angle_state.unsqueeze(1))

            d_t = decay[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            dt_t = dt[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            trap_t = trap[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            now = torch.einsum("brhp,brhn->brhpn", x_r[:, t], k_t)
            prev = torch.einsum("brhp,brhn->brhpn", prev_v, prev_k)
            state = d_t * state + dt_t * ((1.0 - trap_t) * now + trap_t * d_t * prev)

            y_t = torch.einsum("brhpn,brhn->brhp", state, q_t)
            y_t = y_t + self.D.view(1, 1, self.nheads, 1) * x_r[:, t]
            y_t = y_t * F.silu(z_r[:, t])
            y_t = torch.einsum("brhp,hrp->bhp", y_t, self.mimo_o.to(y_t.dtype))
            ys.append(rearrange(y_t, "b h p -> b (h p)"))
            prev_k, prev_v = k_t, x_r[:, t]
        return torch.stack(ys, dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, **kwargs):
        device = device or self.in_proj.weight.device
        dtype = dtype or self.in_proj.weight.dtype
        return {
            "angle": torch.zeros(batch_size, self.nheads, self.num_rope_angles, device=device, dtype=torch.float32),
            "state": torch.zeros(
                batch_size,
                self.mimo_rank,
                self.nheads,
                self.headdim,
                self.d_state,
                device=device,
                dtype=dtype,
            ),
        }


class TrapezoidalSSM(Mamba3Core):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_rope=False, is_mimo=False, **kwargs)


class ComplexRoPESSM(Mamba3Core):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_rope=True, is_mimo=False, **kwargs)


class MIMOSSMBlock(Mamba3Core):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_rope=True, is_mimo=True, **kwargs)


def _build_official_mamba3(
    *,
    d_model: int,
    d_state: int,
    expand: int,
    headdim: int,
    ngroups: int,
    variant: str,
    mimo_rank: int,
    layer_idx=None,
    device=None,
    dtype=None,
    **kwargs,
) -> nn.Module:
    """Build the upstream Mamba-3 block when the CUDA extension is available."""
    if OfficialMamba3 is None:
        raise ImportError(f"Official Mamba3 is unavailable: {_OFFICIAL_MAMBA3_ERROR}")
    if variant == "trapezoidal":
        raise ValueError("The official module exposes full Mamba-3; use the reference backend for trapezoidal-only ablations.")
    return OfficialMamba3(
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        headdim=headdim,
        ngroups=ngroups,
        is_mimo=(variant == "mimo"),
        mimo_rank=mimo_rank,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype,
        **kwargs,
    )


MAMBA3_VARIANTS = {
    "trapezoidal": TrapezoidalSSM,
    "complex": ComplexRoPESSM,
    "mimo": MIMOSSMBlock,
}


class BiMamba3(nn.Module):
    """Bidirectional Mamba-3: forward + backward scan, outputs summed."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        variant: str = "complex",
        mimo_rank: int = 4,
        impl: str = "auto",
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        if variant not in MAMBA3_VARIANTS:
            raise ValueError(f"Unknown Mamba-3 variant '{variant}'. Expected {list(MAMBA3_VARIANTS)}")
        cfg = dict(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            mimo_rank=mimo_rank,
            device=device,
            dtype=dtype,
        )
        cfg.update(kwargs)
        if impl not in {"auto", "official", "reference"}:
            raise ValueError("impl must be one of: auto, official, reference")

        self.impl = "reference"
        if impl in {"auto", "official"} and variant != "trapezoidal":
            try:
                self.fwd = _build_official_mamba3(variant=variant, layer_idx=layer_idx, **cfg)
                self.bwd = _build_official_mamba3(variant=variant, layer_idx=layer_idx, **cfg)
                self.impl = "official"
            except Exception:
                if impl == "official":
                    raise
                cls = MAMBA3_VARIANTS[variant]
                self.fwd = cls(**cfg)
                self.bwd = cls(**cfg)
        else:
            cls = MAMBA3_VARIANTS[variant]
            self.fwd = cls(**cfg)
            self.bwd = cls(**cfg)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        out_fwd = self.fwd(hidden_states)
        out_bwd = self.bwd(hidden_states.flip(dims=[1])).flip(dims=[1])
        return out_fwd + out_bwd

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            "fwd": self.fwd.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs),
            "bwd": self.bwd.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs),
        }


if __name__ == "__main__":
    batch, length, dim = 2, 64, 128
    x = torch.randn(batch, length, dim)
    for variant in ["trapezoidal", "complex", "mimo"]:
        model = BiMamba3(dim, d_state=32, expand=2, headdim=32, variant=variant, mimo_rank=2)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"BiMamba3[{variant:12s}] | in={tuple(x.shape)} -> out={tuple(out.shape)} | params={n_params:,}")
