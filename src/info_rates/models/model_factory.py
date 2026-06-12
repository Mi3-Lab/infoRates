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
    def _interpolate_pos_embed(native_state: dict, target_model, native_size: int, target_size: int) -> dict:
        """Copy weights from native-size state dict into target model, bicubic-interpolating pos_embed.

        Handles three layouts found in HuggingFace video transformers:
          [1, h*w+1, D]   — space pos_embed with cls token (TimeSformer, ViViT)
          [1, h*w,   D]   — space pos_embed without cls token
          [1, T*h*w, D]   — tubelet pos_embed, temporal dim T preserved (VideoMAE)
        Any other mismatched tensor falls back to target model's random init.
        """
        h_old = w_old = native_size // 16
        h_new = w_new = target_size // 16
        target_state = target_model.state_dict()
        new_state = {}

        for key in target_state:
            if key not in native_state:
                new_state[key] = target_state[key]
                continue

            native_t = native_state[key]
            target_t = target_state[key]

            if native_t.shape == target_t.shape:
                new_state[key] = native_t
                continue

            # Only attempt interpolation for [1, N, D] tensors
            if native_t.dim() != 3 or native_t.shape[0] != 1:
                new_state[key] = target_t
                continue

            D = native_t.shape[-1]
            n_nat = native_t.shape[1]
            n_tgt = target_t.shape[1]
            src = native_t.float()

            # Case 1: [1, h*w+1, D] — cls token + spatial patches
            if n_nat == h_old * w_old + 1 and n_tgt == h_new * w_new + 1:
                cls = src[:, :1, :]
                spatial = src[:, 1:, :].reshape(1, h_old, w_old, D).permute(0, 3, 1, 2)
                spatial = F.interpolate(spatial, size=(h_new, w_new), mode="bicubic", align_corners=False)
                spatial = spatial.permute(0, 2, 3, 1).reshape(1, h_new * w_new, D)
                new_state[key] = torch.cat([cls, spatial], dim=1).to(target_t.dtype)

            # Case 2: [1, h*w, D] — spatial patches, no cls token
            elif n_nat == h_old * w_old and n_tgt == h_new * w_new:
                spatial = src.reshape(1, h_old, w_old, D).permute(0, 3, 1, 2)
                spatial = F.interpolate(spatial, size=(h_new, w_new), mode="bicubic", align_corners=False)
                new_state[key] = spatial.permute(0, 2, 3, 1).reshape(1, h_new * w_new, D).to(target_t.dtype)

            # Case 3: [1, T*h*w, D] — tubelet layout (VideoMAE), T preserved
            elif (n_nat % (h_old * w_old) == 0 and n_tgt % (h_new * w_new) == 0):
                T = n_nat // (h_old * w_old)
                if T == n_tgt // (h_new * w_new):
                    spatial = src[0].reshape(T, h_old, w_old, D).permute(0, 3, 1, 2)  # [T, D, h, w]
                    spatial = F.interpolate(spatial, size=(h_new, w_new), mode="bicubic", align_corners=False)
                    spatial = spatial.permute(0, 2, 3, 1).reshape(1, T * h_new * w_new, D)
                    new_state[key] = spatial.to(target_t.dtype)
                else:
                    new_state[key] = target_t
            else:
                new_state[key] = target_t

        return new_state

    @staticmethod
    def load_model(model_name: str, num_labels: int = 101, checkpoint: str | Path | None = None,
                   device: str = "cuda", input_size: int | None = None):
        """Load model from checkpoint or from pretrained HuggingFace weights.

        When input_size differs from the model's native size and we are loading from
        HuggingFace pretrained weights (not a fine-tuned checkpoint), positional embeddings
        are bicubic-interpolated from native resolution to target resolution.  This preserves
        the spatial layout priors from pretraining, avoiding random re-initialization.

        When loading from a fine-tuned checkpoint, the pos_embed is already stored at the
        correct target size, so no interpolation is applied.
        """
        from transformers import AutoConfig
        info = ModelFactory.get_model_info(model_name)
        native_size = info["input_size"]
        target_size = input_size if input_size is not None else native_size

        loading_from_checkpoint = bool(checkpoint and Path(checkpoint).exists())
        src = str(checkpoint) if loading_from_checkpoint else info["model_id"]

        if target_size != native_size and not loading_from_checkpoint:
            # Step 1: load at native size to capture correct pos_embed
            model_native = AutoModelForVideoClassification.from_pretrained(
                src,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
            native_state = model_native.state_dict()
            del model_native

            # Step 2: build target-size architecture
            config = AutoConfig.from_pretrained(src)
            config.image_size = target_size
            config.num_labels = num_labels
            model = AutoModelForVideoClassification.from_pretrained(
                src,
                config=config,
                ignore_mismatched_sizes=True,
            )

            # Step 3: copy weights, bicubic-interpolating pos_embed tensors
            new_state = ModelFactory._interpolate_pos_embed(native_state, model, native_size, target_size)
            model.load_state_dict(new_state, strict=True)

        elif target_size != native_size:
            # Loading from a checkpoint already stored at target_size — load directly
            config = AutoConfig.from_pretrained(src)
            config.image_size = target_size
            config.num_labels = num_labels
            model = AutoModelForVideoClassification.from_pretrained(
                src,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            model = AutoModelForVideoClassification.from_pretrained(
                src,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )

        model = model.to(device)
        return model, info
