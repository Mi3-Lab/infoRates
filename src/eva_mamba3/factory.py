"""Convenience factory functions for standard EVA model configurations.

Each function returns an EVAWrapper ready for training or evaluation.
"""
from __future__ import annotations

from .eva     import EVATokenizer
from .wrapper import EVAWrapper


def build_eva_mamba3(
    n_classes:      int,
    C:              int   = 512,
    D:              int   = 384,
    n_layers:       int   = 12,
    d_state:        int   = 64,
    freeze_encoder: bool  = True,
) -> EVAWrapper:
    """EVA + ResNet-50 spatial encoder + Bidirectional Mamba-3 temporal backbone.

    Primary model for the paper.
    """
    from .encoders import ResNetFrameEncoder
    from .backbone import Mamba3Backbone

    encoder  = ResNetFrameEncoder(C_out=C, freeze=freeze_encoder)
    eva      = EVATokenizer(C=C, D=D)
    backbone = Mamba3Backbone(D_in=D, D=D, n_layers=n_layers,
                              d_state=d_state, n_classes=n_classes)
    return EVAWrapper(encoder, eva, backbone)


def build_uniform_mamba3(
    n_classes: int,
    C: int = 512,
    D: int = 384,
    n_layers: int = 12,
    d_state: int = 64,
    freeze_encoder: bool = True,
):
    """Uniform stride + ResNet-50 + Mamba-3 baseline."""
    from .baselines import UniformStrideWrapper
    from .encoders import ResNetFrameEncoder
    from .backbone import Mamba3Backbone

    encoder = ResNetFrameEncoder(C_out=C, freeze=freeze_encoder)
    backbone = Mamba3Backbone(
        D_in=D,
        D=D,
        n_layers=n_layers,
        d_state=d_state,
        n_classes=n_classes,
    )
    return UniformStrideWrapper(encoder, backbone, C_in=C, D=D)


def build_uniform_transformer(
    n_classes: int,
    C: int = 512,
    D: int = 384,
    n_layers: int = 4,
    n_heads: int = 8,
    freeze_encoder: bool = True,
):
    """Uniform stride + ResNet-50 + temporal Transformer baseline."""
    from .baselines import UniformStrideWrapper
    from .encoders import ResNetFrameEncoder
    from .backbone import TemporalTransformerBackbone

    encoder = ResNetFrameEncoder(C_out=C, freeze=freeze_encoder)
    backbone = TemporalTransformerBackbone(
        D_in=D,
        D=D,
        n_layers=n_layers,
        n_heads=n_heads,
        n_classes=n_classes,
    )
    return UniformStrideWrapper(encoder, backbone, C_in=C, D=D)


def build_eva_vit_transformer(
    n_classes:      int,
    vit_name:       str  = "vit_base_patch16_224",
    D:              int  = 384,
    n_layers:       int  = 4,
    n_heads:        int  = 8,
    freeze_encoder: bool = True,
) -> EVAWrapper:
    """EVA + ViT-B/16 spatial encoder (timm) + Temporal Transformer backbone.

    Ablation: shows EVA works with ViT-family spatial encoders.
    Requires: pip install timm
    """
    from .encoders import ViTFrameEncoder
    from .backbone import TemporalTransformerBackbone

    encoder  = ViTFrameEncoder(model_name=vit_name, freeze=freeze_encoder)
    eva      = EVATokenizer(C=encoder.C_out, D=D)
    backbone = TemporalTransformerBackbone(D_in=D, D=D, n_layers=n_layers,
                                           n_heads=n_heads, n_classes=n_classes)
    return EVAWrapper(encoder, eva, backbone)


def build_eva_timesformer(
    n_classes: int,
    model_name: str = "facebook/timesformer-base-finetuned-k400",
    D: int = 384,
    n_layers: int = 4,
    n_heads: int = 8,
    freeze_encoder: bool = True,
) -> EVAWrapper:
    """EVA + pretrained TimeSformer feature encoder + temporal Transformer head.

    This is the recommended first experimental path: it starts from a mature
    video-pretrained backbone instead of training Mamba-3 from scratch.
    """
    from .encoders import TimeSformerFrameEncoder
    from .backbone import TemporalTransformerBackbone

    encoder = TimeSformerFrameEncoder(model_name=model_name, freeze=freeze_encoder)
    eva = EVATokenizer(C=encoder.C_out, D=D)
    backbone = TemporalTransformerBackbone(
        D_in=D,
        D=D,
        n_layers=n_layers,
        n_heads=n_heads,
        n_classes=n_classes,
    )
    return EVAWrapper(encoder, eva, backbone)


def build_timesformer_baseline(
    n_classes: int,
    model_name: str = "facebook/timesformer-base-finetuned-k400",
    num_frames: int = 8,
    **_,
):
    """Pure TimeSformer baseline with stride-controlled frame sampling."""
    from .baselines import TimeSformerClassifierWrapper

    return TimeSformerClassifierWrapper(
        n_classes=n_classes,
        model_name=model_name,
        num_frames=num_frames,
    )


def build_eva_videomae_transformer(
    n_classes:      int,
    model_name:     str  = "MCG-NJU/videomae-base",
    D:              int  = 384,
    n_layers:       int  = 4,
    n_heads:        int  = 8,
    freeze_encoder: bool = True,
) -> EVAWrapper:
    """EVA + VideoMAE spatial encoder (HuggingFace) + Temporal Transformer backbone.

    Ablation: shows EVA works with VideoMAE-family encoders.
    Requires: pip install transformers
    """
    from .encoders import VideoMAEFrameEncoder
    from .backbone import TemporalTransformerBackbone

    encoder  = VideoMAEFrameEncoder(model_name=model_name, freeze=freeze_encoder)
    eva      = EVATokenizer(C=encoder.C_out, D=D)
    backbone = TemporalTransformerBackbone(D_in=D, D=D, n_layers=n_layers,
                                           n_heads=n_heads, n_classes=n_classes)
    return EVAWrapper(encoder, eva, backbone)


# ── Registry ─────────────────────────────────────────────────────────────────
# Maps --backbone CLI flag → factory function
BACKBONE_REGISTRY: dict[str, callable] = {
    "timesformer_baseline": build_timesformer_baseline,
    "eva_timesformer":     build_eva_timesformer,
    "mamba3":             build_eva_mamba3,
    "uniform_mamba3":     build_uniform_mamba3,
    "uniform_transformer": build_uniform_transformer,
    "vit_transformer":    build_eva_vit_transformer,
    "videomae_transformer": build_eva_videomae_transformer,
}
