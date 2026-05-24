"""TorchVision video backbones for non-transformer ACCV baselines."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision.models.video import (
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r2plus1d_18,
    r3d_18,
)


MODEL_REGISTRY = {
    "r3d_18": {
        "builder": r3d_18,
        "weights": R3D_18_Weights.KINETICS400_V1,
        "default_frames": 16,
        "description": "TorchVision 3D ResNet-18",
    },
    "mc3_18": {
        "builder": mc3_18,
        "weights": MC3_18_Weights.KINETICS400_V1,
        "default_frames": 16,
        "description": "TorchVision mixed-convolution 3D ResNet-18",
    },
    "r2plus1d_18": {
        "builder": r2plus1d_18,
        "weights": R2Plus1D_18_Weights.KINETICS400_V1,
        "default_frames": 16,
        "description": "TorchVision R(2+1)D-18",
    },
}


class TensorBatch(dict):
    """Small BatchFeature-like dict so the shared evaluator can call `.to()`."""

    def to(self, device):
        return TensorBatch({k: v.to(device) if torch.is_tensor(v) else v for k, v in self.items()})


class TorchvisionVideoProcessor:
    """Convert RGB numpy frames to normalized `B,C,T,H,W` tensors."""

    def __init__(
        self,
        size: int = 112,
        mean: tuple[float, float, float] = (0.43216, 0.394666, 0.37645),
        std: tuple[float, float, float] = (0.22803, 0.22145, 0.216989),
    ):
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, videos, return_tensors: str = "pt") -> TensorBatch:
        if videos and isinstance(videos[0], np.ndarray):
            videos = [videos]
        tensors = []
        for frames in videos:
            arr = np.stack(frames, axis=0).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
            tensor = (tensor - self.mean) / self.std
            tensors.append(tensor)
        return TensorBatch({"pixel_values": torch.stack(tensors, dim=0)})


class TorchvisionVideoClassifier(nn.Module):
    """Adapter that exposes a Hugging Face-like `.logits` output."""

    def __init__(self, model: nn.Module, num_labels: int):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(num_labels=num_labels, id2label={i: str(i) for i in range(num_labels)})

    def forward(self, pixel_values: torch.Tensor):
        return SimpleNamespace(logits=self.model(pixel_values))


def create_torchvision_video_model(
    model_name: str,
    num_labels: int,
    pretrained: bool = True,
) -> TorchvisionVideoClassifier:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown TorchVision video model: {model_name}")
    info = MODEL_REGISTRY[model_name]
    weights = info["weights"] if pretrained else None
    model = info["builder"](weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)
    return TorchvisionVideoClassifier(model, num_labels=num_labels)


def save_torchvision_video_checkpoint(
    save_dir: str | Path,
    model: TorchvisionVideoClassifier,
    model_name: str,
    class_names: list[str],
    num_frames: int,
    input_size: int,
    extra: dict | None = None,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), save_dir / "model.pt")
    config = {
        "backend": "torchvision_video",
        "model_name": model_name,
        "num_labels": len(class_names),
        "class_names": class_names,
        "num_frames": num_frames,
        "input_size": input_size,
    }
    if extra:
        config.update(extra)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_torchvision_video_checkpoint(checkpoint: str | Path, device: str = "cuda"):
    checkpoint = Path(checkpoint)
    with open(checkpoint / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("backend") != "torchvision_video":
        raise ValueError(f"Not a TorchVision video checkpoint: {checkpoint}")
    model = create_torchvision_video_model(
        config["model_name"],
        num_labels=int(config["num_labels"]),
        pretrained=False,
    )
    state = torch.load(checkpoint / "model.pt", map_location="cpu")
    model.model.load_state_dict(state)
    model.to(device)
    model.eval()
    processor = TorchvisionVideoProcessor(size=int(config.get("input_size", 112)))
    return model, processor, config
