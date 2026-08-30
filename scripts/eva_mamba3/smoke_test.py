"""Smoke test for EVA-Mamba3 (modular EVAWrapper architecture).

Tests:
  1. EVATokenizer output shapes at all strides (M=1 and M=2)
  2. Generic EVAWrapper interface with a mock encoder and mock backbone
  3. EVA-Mamba3 (full model via factory) forward pass
  4. EVALoss backward — all 5 terms finite
  5. Eval mode: no dense forward pass

Run with:
    TORCH_CUDA_ARCH_LIST="12.0" python scripts/eva_mamba3/smoke_test.py
"""
import os, sys
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn
from eva_mamba3.eva     import EVATokenizer
from eva_mamba3.wrapper import EVAWrapper
from eva_mamba3.losses  import EVALoss
from eva_mamba3.factory import build_eva_mamba3, build_uniform_transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B  = 2
T  = 32    # divisible by all strides {1,2,4,8,16}
H  = W = 64
C  = 128   # tiny encoder channels
D  = 64    # tiny backbone width
N_CLS  = 10
N_LAY  = 2

STRIDES = [1, 2, 4, 8, 16]


# ── 1. EVATokenizer shapes ────────────────────────────────────────────────────

def test_eva_shapes():
    print("=== 1. EVATokenizer shape test ===")
    eva = EVATokenizer(C=C, D=D).to(DEVICE)
    f   = torch.randn(B, T, C, device=DEVICE)
    for s in STRIDES:
        for M in (1, 2):
            if M == 2 and s < 8:
                continue
            z = eva(f, s=s, M=M)
            assert z.shape == (B, T // s * M, D), f"s={s} M={M}: {z.shape}"
            print(f"  s={s:2d}  M={M}  Z{tuple(z.shape)}  ✓")
    print()


# ── 2. Generic EVAWrapper with mock encoder/backbone ─────────────────────────

class MockFrameEncoder(nn.Module):
    """Minimal frame encoder — linear projection of flattened frames."""
    def __init__(self, C_out):
        super().__init__()
        self.C_out = C_out
        self.proj  = nn.Linear(H * W * 3, C_out)
    def forward(self, frames):
        B_, T_, H_, W_, _ = frames.shape
        x = frames.view(B_, T_, -1).float() / 255.0
        return self.proj(x)   # (B, T, C_out)

class MockBackbone(nn.Module):
    """Minimal temporal backbone — mean pool + linear head."""
    def __init__(self, D_in, n_classes):
        super().__init__()
        self.head = nn.Linear(D_in, n_classes)
    def forward(self, Z):
        return self.head(Z.mean(1))   # (B, n_classes)

def test_generic_wrapper():
    print("=== 2. Generic EVAWrapper (mock encoder + backbone) ===")
    encoder  = MockFrameEncoder(C_out=C).to(DEVICE)
    eva      = EVATokenizer(C=C, D=D).to(DEVICE)
    backbone = MockBackbone(D_in=D, n_classes=N_CLS).to(DEVICE)
    model    = EVAWrapper(encoder, eva, backbone).to(DEVICE)
    model.train()

    frames = torch.randint(0, 255, (B, T, H, W, 3), dtype=torch.uint8, device=DEVICE)
    for s in STRIDES:
        out = model(frames, s=s)
        assert out["logits"].shape == (B, N_CLS)
        if s > 1:
            assert out["logits_dense"] is not None
        print(f"  s={s:2d}  logits{tuple(out['logits'].shape)}  ✓")
    print()
    return model, encoder


# ── 3. EVA-Mamba3 via factory ─────────────────────────────────────────────────

def test_factory_mamba3():
    print("=== 3. build_eva_mamba3 (factory) ===")
    model = build_eva_mamba3(n_classes=N_CLS, C=C, D=D, n_layers=N_LAY).to(DEVICE)
    model.train()
    frames = torch.randint(0, 255, (B, T, H, W, 3), dtype=torch.uint8, device=DEVICE)
    for s in STRIDES:
        out = model(frames, s=s)
        assert out["logits"].shape == (B, N_CLS)
        print(f"  s={s:2d}  logits{tuple(out['logits'].shape)}"
              f"  a_evt{tuple(out['a_evt'].shape)}  ✓")
    print()
    return model


# ── 4. EVALoss backward ───────────────────────────────────────────────────────

def test_loss_backward(model):
    print("=== 4. EVALoss backward (all 5 terms) ===")
    criterion = EVALoss()
    model.train()
    frames = torch.randint(0, 255, (B, T, H, W, 3), dtype=torch.uint8, device=DEVICE)
    labels = torch.randint(0, N_CLS, (B,), device=DEVICE)

    for s in STRIDES:
        model.zero_grad()
        out = model(frames, s=s)
        out2 = model(torch.roll(frames, -1, 1), s=s) if s > 1 else None
        ld   = criterion(out, labels, s, out2)
        assert torch.isfinite(ld["total"]), f"s={s}: non-finite loss"
        ld["total"].backward()
        print(f"  s={s:2d}  keys={list(ld.keys())}  total={ld['total'].item():.4f}  ✓")
    print()


# ── 5. Eval mode: no dense forward ───────────────────────────────────────────

def test_eval_mode(model):
    print("=== 5. Eval mode — logits_dense must be None ===")
    model.eval()
    frames = torch.randint(0, 255, (B, T, H, W, 3), dtype=torch.uint8, device=DEVICE)
    with torch.no_grad():
        for s in STRIDES:
            out = model(frames, s=s)
            assert out["logits_dense"] is None, f"s={s}: dense path ran in eval"
            print(f"  s={s:2d}  logits_dense=None  ✓")
    print()


def test_uniform_baseline():
    print("=== 7. Uniform stride baseline ===")
    model = build_uniform_transformer(
        n_classes=N_CLS,
        C=C,
        D=D,
        n_layers=1,
        n_heads=4,
    ).to(DEVICE)
    model.train()
    frames = torch.randint(0, 255, (B, T, H, W, 3), dtype=torch.uint8, device=DEVICE)
    labels = torch.randint(0, N_CLS, (B,), device=DEVICE)
    criterion = EVALoss()

    for s in STRIDES:
        out = model(frames, s=s)
        assert out["logits"].shape == (B, N_CLS)
        ld = criterion(out, labels, s)
        assert set(ld) == {"cls", "total"}
        assert torch.isfinite(ld["total"])
        print(f"  s={s:2d}  logits{tuple(out['logits'].shape)}  loss={ld['total'].item():.4f}  ✓")
    print()


# ── 6. unfreeze_encoder works for both mock and real encoders ─────────────────

def test_unfreeze(wrapper, encoder):
    print("=== 6. unfreeze_encoder ===")
    # Freeze first
    for p in encoder.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in encoder.parameters())

    wrapper.unfreeze_encoder()
    assert any(p.requires_grad for p in encoder.parameters())
    print("  unfreeze_encoder()  ✓")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
    print()

    test_eva_shapes()
    wrapper, encoder = test_generic_wrapper()
    test_loss_backward(wrapper)
    test_eval_mode(wrapper)
    test_unfreeze(wrapper, encoder)

    mamba3_model = test_factory_mamba3()
    test_loss_backward(mamba3_model)
    test_eval_mode(mamba3_model)
    test_uniform_baseline()

    print("=" * 45)
    print("All smoke tests passed.")
