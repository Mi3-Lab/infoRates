#!/usr/bin/env python3
"""Express the stride axis in seconds instead of frames (reviewer vdwp).

vdwp's objection:

    "The frame stride is often an issue for action recognition experiments, but
     it is related to frame rate (fps) of videos. Many datasets contain videos of
     different fps (from 25 to 60). A static number of frame stride doesn't make
     sense for those videos as a single frame means different temporal length."

Correct, and worse than stated: the fps table the duration analysis relies on
(e10_clip_duration.py) is hardcoded, approximate, and wrong for three datasets.
Measured from the actual files: HMDB-51 is 30 not 25, AUTSL is 30 not 25,
EPIC-Kitchens is 50 not 60, and FineGym is missing from the dict entirely so it
silently defaults to 25.

The physically meaningful quantity is not stride but the effective temporal
sampling interval actually delivered to the model:

    interval_seconds = (source_frames / fps) / k_distinct

where k_distinct is the number of distinct frames the sampler produced, not the
nominal budget. Because select_frame_indices re-uniformizes the candidate pool,
stride s only changes k once ceil(window/s) < budget -- so two datasets at the
same nominal stride can sit an order of magnitude apart in seconds per sample.

This script pairs measured per-clip sampling intervals with accuracy so the
x-axis can be redrawn in seconds, and reports whether the per-dataset curves
collapse onto a common one once expressed that way.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_seconds_axis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "evaluations/accv2026/coverage_stride_sweep"
OUT = ROOT / "evaluations/accv2026/rebuttal"

# Measured with decord on the actual validation files (120-clip sample each),
# not the hardcoded approximations in e10_clip_duration.py.
FPS_MEASURED = {"ucf101": 25.0, "ssv2": 12.0, "hmdb51": 30.0, "diving48": 25.0,
                "autsl": 30.0, "driveact": 15.0, "epic_kitchens": 50.0}
FPS_ASSUMED = {"ucf101": 25, "ssv2": 12, "hmdb51": 25, "diving48": 25,
               "autsl": 25, "driveact": 15, "epic_kitchens": 60}

MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae"]
STRIDES = [1, 2, 4, 8, 16]
SUFFIXES = ("", "_trainres224", "_res224")


def load(model: str, dataset: str, stride: int) -> pd.DataFrame | None:
    for suffix in SUFFIXES:
        f = SWEEP / f"{model}_{dataset}{suffix}" / f"cov100_s{stride}_samples.csv"
        if f.exists():
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            return df[df["error"].isna() & ~df["skipped"].astype(bool)]
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Measured vs assumed frame rate")
    print("=" * 78)
    print(f"{'dataset':<15s}{'assumed':>9s}{'measured':>10s}{'error':>9s}")
    for d, fps in FPS_MEASURED.items():
        a = FPS_ASSUMED[d]
        flag = "" if abs(a - fps) < 0.5 else "  <-- wrong in S6"
        print(f"{d:<15s}{a:9d}{fps:10.1f}{100 * (a - fps) / fps:8.0f}%{flag}")

    rows = []
    for dataset, fps in FPS_MEASURED.items():
        for model in MODELS:
            for stride in STRIDES:
                df = load(model, dataset, stride)
                if df is None or df.empty:
                    continue
                k = df.candidate_frames.clip(upper=df.model_input_frames)
                # Guard against the degenerate k=0 rows a failed decode can leave.
                valid = k > 0
                if not valid.any():
                    continue
                interval = (df.source_frames[valid] / fps) / k[valid]
                rows.append(dict(
                    dataset=dataset, model=model, stride=stride, fps=fps,
                    top1=100 * df.correct_top1.mean(),
                    median_interval_s=float(np.median(interval)),
                    median_k=float(np.median(k[valid])),
                    median_duration_s=float(np.median(df.source_frames / fps)),
                ))
    if not rows:
        print("\nno per-clip data found")
        return

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "seconds_axis.csv", index=False)

    print("\n" + "=" * 78)
    print("Seconds between sampled frames at each nominal stride (median over")
    print("models and clips) -- the same stride means very different sampling")
    print("=" * 78)
    piv = res.pivot_table(index="dataset", columns="stride",
                          values="median_interval_s", aggfunc="median")
    print(piv.round(3).to_string())

    print("\nSpread across datasets at a fixed nominal stride:")
    for s in STRIDES:
        col = piv[s].dropna()
        if len(col) > 1:
            print(f"  stride {s:>2d}: {col.min():.3f}s to {col.max():.3f}s "
                  f"({col.max() / col.min():.1f}x)")

    # Does accuracy track seconds-per-sample better than it tracks stride?
    print("\n" + "=" * 78)
    print("Which axis explains accuracy better, within each dataset?")
    print("=" * 78)
    print(f"{'dataset':<15s}{'rho(top1, stride)':>20s}{'rho(top1, interval)':>22s}")
    for dataset, g in res.groupby("dataset"):
        r_stride = spearmanr(g.stride, g.top1)[0]
        r_int = spearmanr(g.median_interval_s, g.top1)[0]
        print(f"{dataset:<15s}{r_stride:20.3f}{r_int:22.3f}")

    print(f"\nWrote {OUT / 'seconds_axis.csv'}")


if __name__ == "__main__":
    main()
