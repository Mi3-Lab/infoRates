"""Backward-compatibility shim.

New code should use EVAWrapper directly via factory functions:
    from eva_mamba3.factory import build_eva_mamba3
    model = build_eva_mamba3(n_classes=97)

EVAMamba3() here is a thin convenience alias kept for existing call sites.
"""
from .wrapper import EVAWrapper
from .factory import build_eva_mamba3


def EVAMamba3(
    C: int = 512,
    D: int = 384,
    n_classes: int = 97,
    n_layers: int = 12,
    d_state: int = 64,
    freeze_encoder: bool = True,
) -> EVAWrapper:
    return build_eva_mamba3(
        n_classes=n_classes, C=C, D=D,
        n_layers=n_layers, d_state=d_state,
        freeze_encoder=freeze_encoder,
    )
