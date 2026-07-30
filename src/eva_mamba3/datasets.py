"""Dense video dataset for EVA-Mamba3.

Returns T=64 consecutive frames (uint8, HWC) decoded from a random temporal
window within the annotated clip. EVA handles all subsampling internally.

Reuses the existing split loaders from src/info_rates/data/datasets.py.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ── Repo root so we can import info_rates loaders ─────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from info_rates.data.datasets import _LOADERS, load_dataset   # type: ignore

# ── Constants ─────────────────────────────────────────────────────────────────
T_DENSE = 64    # number of dense frames to decode per clip
SIZE    = 224   # spatial resolution

# Dataset → (loader_fn, data_root_relative_to_repo)
# Mirrors DATASET_REGISTRY but exposes the roots directly.
_ROOTS: dict[str, str] = {
    name: root for name, (_, root) in _LOADERS.items()
    if name in {"ucf101", "hmdb51", "diving48", "autsl", "ssv2", "finegym"}
}


# ── Video decoder ──────────────────────────────────────────────────────────────

def _decode_frames(
    path: str,
    n_frames: int = T_DENSE,
    size: int = SIZE,
    random_offset: bool = True,
) -> np.ndarray:
    """Decode n_frames consecutive frames from a video file.

    Returns: (n_frames, size, size, 3) uint8 RGB.
    Last frame is repeated if the video is shorter than n_frames.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total <= 0:
        # Fallback: count by reading
        frames_buf = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            frames_buf.append(cv2.resize(frm, (size, size)))
        cap.release()
        if not frames_buf:
            return np.zeros((n_frames, size, size, 3), dtype=np.uint8)
        total = len(frames_buf)
        start = random.randint(0, max(0, total - n_frames)) if random_offset else 0
        raw = frames_buf[start: start + n_frames]
    else:
        start = random.randint(0, max(0, total - n_frames)) if random_offset else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        raw = []
        for _ in range(n_frames):
            ret, frm = cap.read()
            if not ret:
                break
            raw.append(cv2.resize(frm, (size, size)))
        cap.release()

    # Pad with last frame if short
    if raw:
        while len(raw) < n_frames:
            raw.append(raw[-1].copy())
    else:
        return np.zeros((n_frames, size, size, 3), dtype=np.uint8)

    frames = np.stack(raw[:n_frames])              # (T, H, W, 3) BGR
    frames = frames[..., ::-1].copy()              # BGR → RGB
    return frames.astype(np.uint8)


# ── Augmentation helpers ───────────────────────────────────────────────────────

def _random_resized_crop(
    frames: np.ndarray,
    size: int = SIZE,
    scale: Tuple[float, float] = (0.7, 1.0),
) -> np.ndarray:
    """Spatially consistent random resized crop applied to all frames."""
    H, W = frames.shape[1], frames.shape[2]
    area = H * W
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(3 / 4, 4 / 3)
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if w <= W and h <= H:
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)
            frames = frames[:, y:y+h, x:x+w]
            out = np.stack([cv2.resize(f, (size, size)) for f in frames])
            return out
    # Fallback: centre crop
    s = min(H, W)
    y, x = (H - s) // 2, (W - s) // 2
    frames = frames[:, y:y+s, x:x+s]
    return np.stack([cv2.resize(f, (size, size)) for f in frames])


def _horizontal_flip(frames: np.ndarray) -> np.ndarray:
    return frames[:, :, ::-1].copy()


# ── Dataset ────────────────────────────────────────────────────────────────────

class DenseVideoDataset(Dataset):
    """Returns (frames, label) where frames is (T, H, W, 3) uint8 RGB.

    Args:
        dataset_name: one of {ucf101, hmdb51, diving48, autsl, ssv2, finegym}
        split:        'train' or 'val'
        T:            number of dense frames to decode (default 64)
        augment:      apply random crop + flip (train only)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        T: int = T_DENSE,
        augment: bool = True,
        data_root: str | None = None,
    ):
        assert dataset_name in _LOADERS, \
            f"Unknown dataset '{dataset_name}'. Available: {list(_LOADERS)}"
        self.T       = T
        self.augment = augment and (split == "train")

        _, default_root = _LOADERS[dataset_name]
        root = data_root or str(_ROOT / default_root)
        class_names, train_files, val_files = load_dataset(dataset_name, root)

        self.class_names: List[str] = class_names
        self.n_classes: int = len(class_names)
        self.files: List[Tuple[str, int]] = train_files if split == "train" else val_files
        if not self.files:
            raise ValueError(
                f"No videos found for dataset={dataset_name!r}, split={split!r}, "
                f"root={root!r}. Check the manifest and local video files."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        path, label = self.files[idx]
        frames = _decode_frames(path, n_frames=self.T, random_offset=self.augment)

        if self.augment:
            frames = _random_resized_crop(frames)
            if random.random() < 0.5:
                frames = _horizontal_flip(frames)
        else:
            # Centre-crop at inference
            H, W = frames.shape[1], frames.shape[2]
            s = min(H, W)
            y, x = (H - s) // 2, (W - s) // 2
            frames = frames[:, y:y+s, x:x+s]
            frames = np.stack([cv2.resize(f, (SIZE, SIZE)) for f in frames])

        # (T, H, W, 3) uint8  →  Tensor
        t = torch.from_numpy(frames.copy())       # (T, H, W, 3) uint8
        return t, label


def build_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    **kwargs,
):
    ds = DenseVideoDataset(dataset_name, split=split, **kwargs)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        **loader_kwargs,
    )
