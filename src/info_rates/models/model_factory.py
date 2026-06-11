"""HuggingFace transformer model registry for ACCV 2026 baselines."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForVideoClassification


TRANSFORMER_REGISTRY = {
    "timesformer": {
        "model_id": "facebook/timesformer-base-finetuned-k400",
        "description": "TimeSformer (Divided Space-Time Attention)",
        "architecture": "Transformer",
        "default_frames": 8,
        "input_size": 224,
    },
    "videomae": {
        "model_id": "MCG-NJU/videomae-base-finetuned-kinetics",
        "description": "VideoMAE (Masked Autoencoder for Video)",
        "architecture": "Transformer",
        "default_frames": 16,
        "input_size": 224,
    },
    "vivit": {
        "model_id": "google/vivit-b-16x2",
        "description": "ViViT (Video Vision Transformer)",
        "architecture": "Transformer",
        "default_frames": 32,
        "input_size": 224,
    },
}


def _interp_pos_embed(model, native_size: int, target_size: int) -> None:
    """Bicubic-interpolate spatial positional embeddings in-place.

    Handles three layouts found across TimeSformer / ViViT / VideoMAE:
      - [1, n_spatial+1, D]  — spatial with CLS token  (TimeSformer)
      - [1, n_spatial,   D]  — spatial without CLS     (some ViViT variants)
      - [1, T*n_spatial, D]  — temporal×spatial flat    (VideoMAE)
    """
    patch_size = 16
    H_nat = native_size // patch_size   # 14 for 224px
    H_tgt = target_size // patch_size   # e.g. 6 for 96px
    n_nat = H_nat * H_nat               # 196
    n_tgt = H_tgt * H_tgt               # e.g. 36

    for pname, param in list(model.named_parameters()):
        if "position_embed" not in pname.lower():
            continue
        data = param.data.float()
        if data.dim() != 3 or data.shape[0] != 1:
            continue
        N, D = data.shape[1], data.shape[2]

        if N == n_nat + 1:
            # Spatial + CLS token (TimeSformer, plain ViT style)
            cls_tok = data[:, :1, :]
            grid = data[:, 1:, :].reshape(1, H_nat, H_nat, D).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(H_tgt, H_tgt), mode="bicubic", align_corners=False)
            spatial = grid.permute(0, 2, 3, 1).reshape(1, n_tgt, D)
            new = torch.cat([cls_tok, spatial], dim=1)

        elif N == n_nat:
            # Spatial only, no CLS (some ViViT variants)
            grid = data.reshape(1, H_nat, H_nat, D).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(H_tgt, H_tgt), mode="bicubic", align_corners=False)
            new = grid.permute(0, 2, 3, 1).reshape(1, n_tgt, D)

        elif N % n_nat == 0:
            # Temporal × spatial flat, no CLS (VideoMAE style)
            T = N // n_nat
            frames = data.reshape(T, H_nat, H_nat, D).permute(0, 3, 1, 2)  # [T, D, H, H]
            frames = F.interpolate(frames, size=(H_tgt, H_tgt), mode="bicubic", align_corners=False)
            new = frames.permute(0, 2, 3, 1).reshape(1, T * n_tgt, D)

        elif (N - 1) % n_nat == 0 and N > n_nat + 1:
            # CLS + temporal × spatial flat (ViViT: 1 + T*196 tokens)
            cls_tok = data[:, :1, :]
            T = (N - 1) // n_nat
            frames = data[:, 1:, :].reshape(T, H_nat, H_nat, D).permute(0, 3, 1, 2)
            frames = F.interpolate(frames, size=(H_tgt, H_tgt), mode="bicubic", align_corners=False)
            spatial = frames.permute(0, 2, 3, 1).reshape(1, T * n_tgt, D)
            new = torch.cat([cls_tok, spatial], dim=1)

        else:
            continue

        param.data = new.to(param.dtype)
        print(f"  [PosEmbed] {pname}: {list(data.shape)} → {list(new.shape)}")


class ModelFactory:
    """Unified loader for HuggingFace transformer video models."""

    REGISTRY = TRANSFORMER_REGISTRY

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        if model_name not in TRANSFORMER_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(TRANSFORMER_REGISTRY)}")
        return TRANSFORMER_REGISTRY[model_name]

    @staticmethod
    def load_processor(model_name: str, input_size: int | None = None, **kwargs):
        info = ModelFactory.get_model_info(model_name)
        use_fast = model_name == "videomae"
        if input_size is not None and input_size != info["input_size"]:
            kwargs["size"] = {"height": input_size, "width": input_size}
            kwargs["crop_size"] = {"height": input_size, "width": input_size}
        return AutoImageProcessor.from_pretrained(info["model_id"], use_fast=use_fast, **kwargs)

    @staticmethod
    def load_model(model_name: str, num_labels: int = 101, checkpoint: str | Path | None = None,
                   device: str = "cuda", input_size: int | None = None):
        """Load model with bicubic pos-embedding interpolation for non-native resolutions.

        At native 224px the weights load without any modification.
        At other resolutions the pretrained model is first loaded at 224px (so all
        weights including positional embeddings are loaded correctly from the HF hub /
        checkpoint), then the spatial positional embeddings are bicubic-interpolated
        to the target patch-grid size.  This avoids the silent random reinitialization
        that occurs with ignore_mismatched_sizes=True.
        """
        info = ModelFactory.get_model_info(model_name)
        src = str(checkpoint) if checkpoint and Path(checkpoint).exists() else info["model_id"]

        native_size = info["input_size"]   # 224
        target_size = input_size if input_size is not None else native_size

        if target_size != native_size:
            # Step 1: load at native 224px so all weights (including pos embed) load correctly
            model = AutoModelForVideoClassification.from_pretrained(
                src,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,   # only head may mismatch for num_labels
            )
            # Step 2: bicubic-interpolate spatial positional embeddings in-place
            print(f"  Interpolating pos embeddings: {native_size}px → {target_size}px")
            _interp_pos_embed(model, native_size, target_size)
            # Step 3: update config so attention layers use correct spatial grid size
            # (e.g. TimeSformer computes num_patch_width = config.image_size // config.patch_size
            #  in its divided_space_time attention; wrong value → reshape crash)
            model.config.image_size = target_size
            # Step 4: update stored image_size in patch embedding submodules.
            # ViViT stores self.image_size as int; VideoMAE converts to (H,W) tuple.
            # Both check input size in forward() against this stored value — still seeing
            # 224 after pos-embed interpolation causes a crash.  Preserve the original type.
            for submod in model.modules():
                if submod is model:
                    continue
                if not hasattr(submod, "image_size"):
                    continue
                old = submod.image_size
                if isinstance(old, (tuple, list)):
                    submod.image_size = (target_size, target_size)
                elif isinstance(old, int) and old == native_size:
                    submod.image_size = target_size
        else:
            model = AutoModelForVideoClassification.from_pretrained(
                src,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )

        model = model.to(device)
        return model, info
