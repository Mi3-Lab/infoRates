"""HuggingFace transformer model registry for ACCV 2026 baselines."""

from __future__ import annotations

from pathlib import Path

import torch
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
    def load_processor(model_name: str, **kwargs):
        info = ModelFactory.get_model_info(model_name)
        use_fast = model_name == "videomae"
        return AutoImageProcessor.from_pretrained(info["model_id"], use_fast=use_fast, **kwargs)

    @staticmethod
    def load_model(model_name: str, num_labels: int = 101, checkpoint: str | Path | None = None, device: str = "cuda"):
        """Load model from checkpoint or from pretrained HuggingFace weights."""
        info = ModelFactory.get_model_info(model_name)
        src = str(checkpoint) if checkpoint and Path(checkpoint).exists() else info["model_id"]
        model = AutoModelForVideoClassification.from_pretrained(
            src,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        model = model.to(device)
        return model, info
