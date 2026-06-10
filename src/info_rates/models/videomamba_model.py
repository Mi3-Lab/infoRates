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
_PRETRAINED_REL = Path("fine_tuned_models") / "videomamba_pretrained" / "videomamba_m16_k400_f8_res224.pth"
_SCRATCH_ROOT = Path("/scratch/wesleyferreiramaia/infoRates")
PRETRAINED_PATH = (
    _SCRATCH_ROOT / _PRETRAINED_REL
    if (_SCRATCH_ROOT / _PRETRAINED_REL).exists()
    else ROOT / _PRETRAINED_REL
)
EMBED_DIM = 576  # medium config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


MAMBA_BUNDLED = ROOT / "third_party" / "videomamba_repo" / "mamba"


def _add_videomamba_to_path():
    # Add patched mamba_ssm FIRST — it has causal_conv1d_cuda API compatibility fix
    # (pure-Python fallback in mamba_simple.py avoids the CUDA kernel version mismatch)
    if str(MAMBA_BUNDLED) not in sys.path:
        sys.path.insert(0, str(MAMBA_BUNDLED))
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
        # causal_conv1d CUDA kernel requires bfloat16 path (matches training autocast)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=pixel_values.is_cuda):
            return _LogitsOutput(self.backbone(pixel_values))

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        raw = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
        torch.save(raw.state_dict(), save_dir / "model.pth")


def _extract_state(ckpt: dict) -> dict:
    """Extract the state dict from a VideoMamba checkpoint.

    The K400 checkpoint already uses the correct BiMamba key format with _b suffixes
    (e.g. layers.N.mixer.A_b_log, conv1d_b, etc). No remapping needed.
    """
    return ckpt.get("model", ckpt)


def build_videomamba(
    num_classes: int,
    num_frames: int = 8,
    pretrained_path: str | Path | None = None,
    img_size: int = 224,
) -> VideoMambaModel:
    """Instantiate VideoMamba middle with a fresh classification head.

    If `pretrained_path` is given, loads K400 weights (trained at 224px).
    When img_size != 224, loads at native 224px first (so vm_load works),
    then bicubic-interpolates pos_embed and copies weights to target-size model.
    """
    import torch.nn.functional as F

    _add_videomamba_to_path()
    from models.videomamba import videomamba_middle, load_state_dict as vm_load  # noqa: PLC0415

    if pretrained_path is not None and img_size != 224:
        # Step 1: load checkpoint at native 224px (vm_load requires matching shapes)
        model_224 = videomamba_middle(num_classes=num_classes, num_frames=num_frames)
        ckpt = torch.load(str(pretrained_path), map_location="cpu")
        ckpt = _extract_state(ckpt)
        vm_load(model_224, ckpt, center=True)
        model_224.head = nn.Linear(EMBED_DIM, num_classes)
        nn.init.trunc_normal_(model_224.head.weight, std=0.02)
        nn.init.zeros_(model_224.head.bias)

        # Step 2: interpolate pos_embed 224px → img_size
        pos = model_224.pos_embed.data              # [1, 197, D]
        cls_tok = pos[:, :1, :]                     # [1, 1, D]
        spatial  = pos[:, 1:, :]                    # [1, 196, D]
        D = spatial.shape[-1]
        h_old = w_old = 14                          # 224 // 16
        h_new = w_new = img_size // 16
        spatial = spatial.reshape(1, h_old, w_old, D).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial.float(), size=(h_new, w_new),
                                mode="bicubic", align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).reshape(1, h_new * w_new, D)
        interp_pos = torch.cat([cls_tok, spatial], dim=1)           # [1, N_new+1, D]
        print(f"[VideoMamba] pos_embed interpolated: 224px→{img_size}px "
              f"(14×14→{h_new}×{h_new} patches)")

        # Step 3: build target-size model and copy all weights
        model_target = videomamba_middle(num_classes=num_classes, num_frames=num_frames,
                                         img_size=img_size)
        state = model_224.state_dict()
        state["pos_embed"] = interp_pos
        model_target.load_state_dict(state, strict=True)
        return VideoMambaModel(model_target)

    # Native 224px (or no pretrained)
    model_raw = videomamba_middle(num_classes=num_classes, num_frames=num_frames,
                                  img_size=img_size)
    if pretrained_path is not None:
        ckpt = torch.load(str(pretrained_path), map_location="cpu")
        ckpt = _extract_state(ckpt)
        vm_load(model_raw, ckpt, center=True)
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
