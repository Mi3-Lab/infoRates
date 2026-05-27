"""VideoMamba (Mamba SSM) model wrapper for ACCV 2026 fixed-budget evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
VIDEOMAMBA_REPO = ROOT / "third_party" / "videomamba_repo" / "videomamba" / "video_sm"
PRETRAINED_PATH = ROOT / "fine_tuned_models" / "videomamba_pretrained" / "videomamba_m16_k400_f8_res224.pth"
EMBED_DIM = 576  # medium config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _add_videomamba_to_path():
    if str(VIDEOMAMBA_REPO) not in sys.path:
        sys.path.insert(0, str(VIDEOMAMBA_REPO))


class VideoMambaProcessor:
    """HuggingFace-compatible processor for VideoMamba.

    Accepts either a single video (list[np.ndarray]) or a batch
    (list[list[np.ndarray]]) and returns {"pixel_values": (B, C, T, H, W)}.
    """

    def __init__(self, size: int = 224):
        self.size = size
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __call__(self, frames_or_batch, return_tensors: str = "pt") -> dict:
        if len(frames_or_batch) == 0:
            return {"pixel_values": torch.zeros(0, 3, 1, self.size, self.size)}
        # Detect single video vs. batch
        if isinstance(frames_or_batch[0], np.ndarray):
            videos = [frames_or_batch]  # wrap single video as batch of 1
        else:
            videos = frames_or_batch

        processed = []
        for frames in videos:
            tensors = []
            for f in frames:
                if f.shape[0] != self.size or f.shape[1] != self.size:
                    f = cv2.resize(f, (self.size, self.size))
                t = torch.from_numpy(f.copy()).float().div(255.0).permute(2, 0, 1)
                t = (t - self._mean) / self._std
                tensors.append(t)
            processed.append(torch.stack(tensors, dim=1))  # (C, T, H, W)

        return {"pixel_values": torch.stack(processed, dim=0)}  # (B, C, T, H, W)

    # No-op: metadata is stored in accv_meta.json / config.json
    def save_pretrained(self, save_dir):
        pass


class _LogitsOutput:
    """Minimal HuggingFace-like output wrapper exposing `.logits`."""
    __slots__ = ("logits",)

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class VideoMambaModel(nn.Module):
    """VideoMamba backbone wrapped to match the eval-benchmark interface.

    `model(pixel_values=...)` returns an object with `.logits`.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        num_labels = backbone.head.out_features
        self.config = SimpleNamespace(
            num_labels=num_labels,
            id2label={i: str(i) for i in range(num_labels)},
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> _LogitsOutput:
        return _LogitsOutput(self.backbone(pixel_values))

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        raw = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
        torch.save(raw.state_dict(), save_dir / "model.pth")


def _remap_bimamba_keys(ckpt: dict) -> dict:
    """Remap flat Mamba keys from the K400 pretrained checkpoint to BiMamba layout.

    Original keys: layers.N.mixer.{A_log,D,in_proj,...}
    BiMamba keys:  layers.N.mixer.fwd.{A_log,...}  (+ .bwd copy)

    The backward branch is initialised with the same weights as the forward branch
    so fine-tuning starts from a sensible bidirectional state.
    """
    state = ckpt.get("model", ckpt)
    new_state = {}
    for k, v in state.items():
        if ".mixer." in k and not any(f".mixer.{d}." in k for d in ("fwd", "bwd")):
            prefix, suffix = k.split(".mixer.", 1)
            new_state[f"{prefix}.mixer.fwd.{suffix}"] = v
            new_state[f"{prefix}.mixer.bwd.{suffix}"] = v.clone()
        else:
            new_state[k] = v
    return new_state


def build_videomamba(
    num_classes: int,
    num_frames: int = 8,
    pretrained_path: str | Path | None = None,
) -> VideoMambaModel:
    """Instantiate VideoMamba middle with a fresh classification head.

    If `pretrained_path` is given, loads K400 weights via the repo's
    `load_state_dict` helper (which deletes the original head), then
    replaces the head with a new Linear(embed_dim, num_classes).
    """
    _add_videomamba_to_path()
    from models.videomamba import videomamba_middle, load_state_dict as vm_load  # noqa: PLC0415

    model_raw = videomamba_middle(num_classes=num_classes, num_frames=num_frames)

    if pretrained_path is not None:
        ckpt = torch.load(str(pretrained_path), map_location="cpu")
        ckpt = _remap_bimamba_keys(ckpt)
        vm_load(model_raw, ckpt, center=True)
        # vm_load deletes head.weight/bias and loads strict=False; rebuild head
        model_raw.head = nn.Linear(EMBED_DIM, num_classes)
        nn.init.trunc_normal_(model_raw.head.weight, std=0.02)
        nn.init.zeros_(model_raw.head.bias)
        print(f"[VideoMamba] Pretrained weights loaded from {pretrained_path}")

    return VideoMambaModel(model_raw)


def load_videomamba_checkpoint(
    save_dir: str | Path,
    device: str = "cuda",
) -> Tuple[VideoMambaModel, VideoMambaProcessor, dict]:
    """Load a fine-tuned VideoMamba checkpoint saved by train_videomamba.py."""
    _add_videomamba_to_path()
    from models.videomamba import videomamba_middle  # noqa: PLC0415

    save_dir = Path(save_dir)
    meta = json.loads((save_dir / "accv_meta.json").read_text())

    num_classes = meta["num_labels"]
    num_frames  = meta.get("num_frames", 8)
    input_size  = meta.get("input_size", 224)

    model_raw = videomamba_middle(num_classes=num_classes, num_frames=num_frames)
    state_dict = torch.load(save_dir / "model.pth", map_location="cpu")
    model_raw.load_state_dict(state_dict)
    model_raw.eval()
    model_raw.to(device)

    return VideoMambaModel(model_raw), VideoMambaProcessor(size=input_size), meta
