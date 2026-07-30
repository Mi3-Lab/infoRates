"""Coverage/stride protocol data for EVA evaluation.

This module mirrors the ACCV paper's fixed-budget frame selection while adding
an EVA-specific mode that decodes the dense pre-decimation windows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - cv2 fallback remains available
    VideoReader = None
    cpu = None

Mode = Literal["baseline", "eva_window"]


def select_frame_indices(
    total_frames: int,
    budget: int,
    coverage: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Select fixed-budget frame indices exactly like the previous paper."""
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    budget = max(1, int(budget))
    coverage = int(np.clip(coverage, 1, 100))
    stride = max(1, int(stride))

    window = max(1, int(round(total_frames * coverage / 100.0)))
    candidates = np.arange(0, window, stride, dtype=np.int64)
    if len(candidates) == 0:
        candidates = np.array([0], dtype=np.int64)

    if len(candidates) >= budget:
        pick = np.linspace(0, len(candidates) - 1, budget).round().astype(np.int64)
        return candidates[pick]

    pad = np.full(budget - len(candidates), candidates[-1], dtype=np.int64)
    return np.concatenate([candidates, pad])


def select_eva_window_indices(
    total_frames: int,
    budget: int,
    coverage: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Select dense windows that EVA compresses into the same token budget.

    For each protocol candidate frame, EVA receives the stride-sized window
    starting at that candidate. If the candidate set is shorter than ``budget``,
    the final real candidate/window is repeated, matching fixed-budget padding.
    """
    anchors = select_frame_indices(total_frames, budget, coverage, stride)
    if len(anchors) == 0:
        return anchors
    window = max(1, int(round(total_frames * coverage / 100.0)))
    max_idx = max(0, min(total_frames, window) - 1)
    offsets = np.arange(max(1, int(stride)), dtype=np.int64)
    indices = anchors[:, None] + offsets[None, :]
    return np.clip(indices, 0, max_idx).reshape(-1).astype(np.int64)


def _read_frames_decord(path: str, indices: np.ndarray, size: int) -> np.ndarray:
    if VideoReader is None or cpu is None:
        raise RuntimeError("decord is unavailable")
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"video has no frames: {path}")
    safe = np.clip(indices, 0, total - 1).astype(np.int64)
    frames = vr.get_batch(safe).asnumpy()
    return np.stack([cv2.resize(frame, (size, size)) for frame in frames], axis=0)


def _read_frames_cv2(path: str, indices: np.ndarray, size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"video has no frames: {path}")

    frames = []
    for idx in np.clip(indices, 0, total - 1).astype(np.int64):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"failed to read frame {idx} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (size, size)))
    cap.release()
    return np.stack(frames, axis=0)


def read_indexed_frames(path: str, indices: np.ndarray, size: int = 224) -> np.ndarray:
    try:
        return _read_frames_decord(path, indices, size)
    except Exception:
        return _read_frames_cv2(path, indices, size)


def frame_count(path: str) -> int:
    if VideoReader is not None and cpu is not None:
        try:
            return len(VideoReader(str(path), ctx=cpu(0)))
        except Exception:
            pass
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total


class ProtocolVideoDataset(Dataset):
    """Manifest dataset for fixed-budget coverage/stride evaluation."""

    def __init__(
        self,
        manifest: str | Path,
        split: str | None = "val",
        mode: Mode = "baseline",
        coverage: int = 100,
        stride: int = 1,
        budget: int = 8,
        size: int = 224,
        max_samples: int = 0,
    ):
        if mode not in {"baseline", "eva_window"}:
            raise ValueError(f"unknown protocol mode: {mode}")
        self.mode = mode
        self.coverage = int(coverage)
        self.stride = int(stride)
        self.budget = int(budget)
        self.size = int(size)

        df = pd.read_csv(manifest)
        if split and split != "all" and "split" in df.columns:
            df = df[df["split"].astype(str) == split].copy()
        if "exists" in df.columns:
            df = df[df["exists"].astype(bool)].copy()
        if max_samples and max_samples > 0:
            df = df.iloc[:max_samples].copy()
        if df.empty:
            raise ValueError(f"empty manifest selection: {manifest}")

        label_col = "label_id" if "label_id" in df.columns else "label"
        self.rows = [
            (str(row["video_path"]), int(row[label_col]))
            for _, row in df.iterrows()
            if Path(str(row["video_path"])).exists()
        ]
        if not self.rows:
            raise ValueError(f"no existing videos in manifest selection: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path, label = self.rows[idx]
        total = frame_count(path)
        if self.mode == "baseline":
            indices = select_frame_indices(total, self.budget, self.coverage, self.stride)
        else:
            indices = select_eva_window_indices(total, self.budget, self.coverage, self.stride)
        frames = read_indexed_frames(path, indices, self.size)
        return torch.from_numpy(frames.copy()), label


def build_protocol_loader(
    manifest: str | Path,
    split: str | None = "val",
    mode: Mode = "baseline",
    coverage: int = 100,
    stride: int = 1,
    budget: int = 8,
    size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int = 0,
) -> DataLoader:
    ds = ProtocolVideoDataset(
        manifest=manifest,
        split=split,
        mode=mode,
        coverage=coverage,
        stride=stride,
        budget=budget,
        size=size,
        max_samples=max_samples,
    )
    kwargs = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )
