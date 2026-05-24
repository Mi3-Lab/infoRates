"""Lightweight temporal-demand metrics for ACCV 2026 experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from info_rates.evaluation.benchmark import decode_video_frames, iter_manifest


@dataclass
class TemporalDemandResult:
    dataset: str
    split: str
    video_id: str
    video_path: str
    label_id: int
    label: str
    label_clean: str
    source_frames: int
    processed_frames: int
    demand_mean_abs_diff: float
    demand_std_abs_diff: float
    demand_p95_abs_diff: float
    demand_motion_fraction: float
    decode_time_s: float
    total_time_s: float
    skipped: bool = False
    error: str = ""


def frame_difference_demand(frames: list[np.ndarray], motion_threshold: float = 0.08) -> dict[str, float]:
    """Compute cheap temporal-demand scores from RGB frames.

    The score is intentionally simple: convert sampled frames to grayscale,
    normalize to [0, 1], then summarize frame-to-frame absolute differences.
    It is a temporal-demand proxy, not a physical frequency estimator.
    """
    if len(frames) < 2:
        return {
            "demand_mean_abs_diff": 0.0,
            "demand_std_abs_diff": 0.0,
            "demand_p95_abs_diff": 0.0,
            "demand_motion_fraction": 0.0,
        }

    gray = []
    for frame in frames:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray.append(frame.astype(np.float32) / 255.0)

    stack = np.stack(gray, axis=0)
    diffs = np.abs(np.diff(stack, axis=0))
    per_step = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    return {
        "demand_mean_abs_diff": float(per_step.mean()),
        "demand_std_abs_diff": float(per_step.std(ddof=0)),
        "demand_p95_abs_diff": float(np.percentile(per_step, 95)),
        "demand_motion_fraction": float((diffs > motion_threshold).mean()),
    }


def compute_temporal_demand(
    manifest: pd.DataFrame,
    output_csv: str | Path,
    split: str | None = None,
    frame_budget: int = 16,
    coverage: int = 100,
    stride: int = 1,
    resize: int = 112,
    max_samples: int = 0,
    samples_per_class: int = 0,
    seed: int = 42,
    motion_threshold: float = 0.08,
) -> pd.DataFrame:
    """Compute per-video temporal-demand scores for a manifest."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = iter_manifest(
        manifest,
        split=split,
        max_samples=max_samples,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    if df.empty:
        raise ValueError("manifest selection is empty")

    rows: list[dict] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="temporal-demand"):
        row_dict = row._asdict()
        t0 = time.perf_counter()
        try:
            frames, meta = decode_video_frames(
                row_dict["video_path"],
                budget=frame_budget,
                coverage=coverage,
                stride=stride,
                resize=resize,
            )
            scores = frame_difference_demand(frames, motion_threshold=motion_threshold)
            rows.append(
                asdict(
                    TemporalDemandResult(
                        dataset=str(row_dict.get("dataset", "")),
                        split=str(row_dict.get("split", "")),
                        video_id=str(row_dict.get("video_id", "")),
                        video_path=str(row_dict["video_path"]),
                        label_id=int(row_dict["label_id"]),
                        label=str(row_dict.get("label", "")),
                        label_clean=str(row_dict.get("label_clean", "")),
                        source_frames=int(meta["source_frames"]),
                        processed_frames=int(meta["processed_frames"]),
                        decode_time_s=float(meta["decode_time_s"]),
                        total_time_s=float(time.perf_counter() - t0),
                        **scores,
                    )
                )
            )
        except Exception as exc:
            rows.append(
                asdict(
                    TemporalDemandResult(
                        dataset=str(row_dict.get("dataset", "")),
                        split=str(row_dict.get("split", "")),
                        video_id=str(row_dict.get("video_id", "")),
                        video_path=str(row_dict["video_path"]),
                        label_id=int(row_dict["label_id"]),
                        label=str(row_dict.get("label", "")),
                        label_clean=str(row_dict.get("label_clean", "")),
                        source_frames=0,
                        processed_frames=0,
                        demand_mean_abs_diff=0.0,
                        demand_std_abs_diff=0.0,
                        demand_p95_abs_diff=0.0,
                        demand_motion_fraction=0.0,
                        decode_time_s=0.0,
                        total_time_s=float(time.perf_counter() - t0),
                        skipped=True,
                        error=str(exc),
                    )
                )
            )

        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    return result


def summarize_demand_by_class(demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-video demand scores to class-level statistics."""
    valid = demand[~demand["skipped"].astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["dataset", "split", "label_id", "label_clean"], dropna=False)
        .agg(
            n=("video_id", "size"),
            mean_source_frames=("source_frames", "mean"),
            mean_demand=("demand_mean_abs_diff", "mean"),
            std_demand=("demand_mean_abs_diff", "std"),
            mean_motion_fraction=("demand_motion_fraction", "mean"),
            p95_demand=("demand_p95_abs_diff", "mean"),
        )
        .reset_index()
    )
