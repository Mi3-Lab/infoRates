"""EVA-Mamba3: Evidence-Preserving Temporal Anti-Aliasing for Video Recognition.

IMPORTANT: set TORCH_CUDA_ARCH_LIST="12.0" before importing on Blackwell GPUs
           (sm_120, e.g. RTX PRO 6000) to ensure mamba3_ssm JIT compiles correctly.

Quick start:
    from eva_mamba3.factory import build_eva_mamba3, build_eva_vit_transformer
    model = build_eva_mamba3(n_classes=97)          # EVA + ResNet-50 + Mamba-3
    model = build_eva_vit_transformer(n_classes=97) # EVA + ViT-B/16 + Transformer
"""
from .eva     import EVATokenizer
from .wrapper import EVAWrapper
from .factory import (
    BACKBONE_REGISTRY,
    build_eva_mamba3,
    build_eva_timesformer,
    build_eva_vit_transformer,
    build_timesformer_baseline,
    build_uniform_mamba3,
    build_uniform_transformer,
)
from .model   import EVAMamba3   # backward-compat alias
