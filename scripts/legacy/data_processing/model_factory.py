#!/usr/bin/env python3
"""
Model Factory for Multi-Model Support

Provides unified interface for loading and configuring different video models:
- TimeSformer
- VideoMAE
- ViViT
"""

from pathlib import Path
from transformers import (
    AutoImageProcessor, 
    AutoModelForVideoClassification,
    AutoConfig
)
import torch

class ModelFactory:
    """Factory for creating and loading video classification models."""
    
    # Model registry with key specifications
    REGISTRY = {
        "timesformer": {
            "model_id": "facebook/timesformer-base-finetuned-k400",
            "description": "TimeSformer (Divided Space-Time Attention)",
            "architecture": "Transformer",
            "expected_frames": 8,
            "input_size": 224,
            "default_frames": 8,
        },
        "videomae": {
            "model_id": "MCG-NJU/videomae-base-finetuned-kinetics",
            "description": "VideoMAE (Masked Autoencoder for Video)",
            "architecture": "Transformer",
            "expected_frames": 16,
            "input_size": 224,
            "default_frames": 16,
        },
        "vivit": {
            "model_id": "google/vivit-b-16x2",
            "description": "ViViT (Vision Transformer for Video)",
            "architecture": "Transformer",
            "expected_frames": 32,
            "input_size": 224,
            "default_frames": 32,
        },
        "slowfast": {
            "model_id": "facebookresearch/slowfast-r50",
            "description": "SlowFast (Two-pathway Networks)",
            "architecture": "3D CNN",
            "expected_frames": 32,
            "input_size": 224,
            "default_frames": 32,
            "slow_path_frames": 8,
            "fast_path_frames": 32,
        },
        "x3d": {
            "model_id": "facebookresearch/x3d-xs",
            "description": "X3D (Expandable 3D Networks)",
            "architecture": "3D CNN",
            "expected_frames": 16,
            "input_size": 224,
            "default_frames": 16,
        },
    }
    
    @staticmethod
    def list_available_models():
        """List all available models."""
        models = []
        for name, info in ModelFactory.REGISTRY.items():
            models.append(f"{name}: {info['description']}")
        return "\n".join(models)
    
    @staticmethod
    def get_model_info(model_name):
        """Get model specifications."""
        if model_name not in ModelFactory.REGISTRY:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(ModelFactory.REGISTRY.keys())}"
            )
        return ModelFactory.REGISTRY[model_name]
    
    @staticmethod
    def load_processor(model_name, **kwargs):
        """Load image processor for a specific model."""
        info = ModelFactory.get_model_info(model_name)
        # Prefer fast processors when available (quiet the "slow processor" warning for VideoMAE)
        if model_name == "videomae":
            processor = AutoImageProcessor.from_pretrained(info["model_id"], use_fast=True, **kwargs)
        else:
            processor = AutoImageProcessor.from_pretrained(info["model_id"], **kwargs)
        return processor
    
    @staticmethod
    def load_model(model_name, num_labels=101, checkpoint=None, device="cuda"):
        """
        Load a model with optional fine-tuned checkpoint.
        
        Args:
            model_name: Name of model ('timesformer', 'videomae', 'vivit')
            num_labels: Number of classes (101 for UCF-101)
            checkpoint: Path to fine-tuned model checkpoint (optional)
            device: Device to load model on ('cuda' or 'cpu')
        
        Returns:
            Loaded model, configuration info
        """
        info = ModelFactory.get_model_info(model_name)
        
        print(f"Loading {info['description']}...")
        
        # Load from checkpoint or from pretrained
        if checkpoint and Path(checkpoint).exists():
            print(f"  → From checkpoint: {checkpoint}")
            model = AutoModelForVideoClassification.from_pretrained(
                checkpoint,
                num_labels=num_labels,
            )
        else:
            print(f"  → From pretrained: {info['model_id']}")
            model = AutoModelForVideoClassification.from_pretrained(
                info["model_id"],
                num_labels=num_labels,
                ignore_mismatched_sizes=True,  # Handle class mismatch
            )
        
        # Update config with label mapping
        if not hasattr(model.config, 'id2label') or model.config.id2label is None:
            id2label = {i: str(i) for i in range(num_labels)}
            label2id = {str(i): i for i in range(num_labels)}
            model.config.id2label = id2label
            model.config.label2id = label2id
        
        model = model.to(device)
        model.eval()
        
        print(f"  ✓ Loaded with {num_labels} classes")
        
        return model, info
    
    @staticmethod
    def get_recommended_frames(model_name, target_frames=None):
        """
        Get recommended number of frames for a model.
        
        Args:
            model_name: Name of model
            target_frames: If provided, use this instead of model default
        
        Returns:
            Number of frames to use
        """
        info = ModelFactory.get_model_info(model_name)
        return target_frames if target_frames else info["default_frames"]
    
    @staticmethod
    def validate_frames(model_name, num_frames):
        """Validate if number of frames is reasonable for a model."""
        info = ModelFactory.get_model_info(model_name)
        expected = info["expected_frames"]
        
        if num_frames < expected:
            print(f"⚠️  {model_name} expects ~{expected} frames, but got {num_frames}")
            print(f"    Model may not perform optimally")
        
        return True


def print_model_summary():
    """Print summary of available models."""
    print("\n" + "="*70)
    print("AVAILABLE VIDEO CLASSIFICATION MODELS")
    print("="*70)
    
    for name, info in ModelFactory.REGISTRY.items():
        print(f"\n{name.upper()}")
        print(f"  Description:      {info['description']}")
        print(f"  Architecture:      {info['architecture']}")
        print(f"  Expected Frames:   {info['expected_frames']}")
        print(f"  Input Size:        {info['input_size']}×{info['input_size']}")
        print(f"  Default Frames:    {info['default_frames']}")
        print(f"  Model ID:          {info['model_id']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Test the factory
    print_model_summary()
    
    print("\nTesting model loading (CPU, no actual download)...")
    for model_name in ["timesformer", "videomae", "vivit"]:
        info = ModelFactory.get_model_info(model_name)
        print(f"✓ {model_name}: {info['description']}")
