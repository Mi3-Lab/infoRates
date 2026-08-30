#!/usr/bin/env python3
"""Is the TDS ranking an artifact of how TDS is defined? (reviewer pxtb)

pxtb calls the metric weak on three specific grounds:

  1. "only the average accuracy difference between stride 1 and stride 16, so it
      ignores the intermediate degradation curve"
  2. "depends strongly on baseline accuracy and floor effects"
  3. "the positive-part operation also discards cases where sparse sampling
      improves performance"

Each is a concrete, testable claim. This script recomputes the dataset ranking
under variants that remove each objection in turn and correlates every variant
against the published TDS. If the ranking is stable across all of them, the
objection is about elegance rather than about the conclusion.

Variants
  published    mean over models of max(acc_s1 - acc_s16, 0)          -- as in Eq.1
  unclipped    same without the positive part                        -- objection 3
  auc          area between the s=1 line and the full stride curve,
               trapezoidal in log2(stride), so every intermediate
               stride contributes                                    -- objection 1
  relative     (acc_s1 - acc_s16) / acc_s1, per model                -- objection 2
  normalized   drop divided by the headroom above chance (1/n_classes),
               which is the sharper form of the floor-effect worry    -- objection 2

Usage:  .venv/bin/python scripts/accv2026/rebuttal_tds_metric_variants.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "evaluations/accv2026/coverage_stride_sweep"
DASH = ROOT / "dashboard/data/sweep_summary.csv"
OUT = ROOT / "evaluations/accv2026/rebuttal"

MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48",
            "finegym", "autsl", "driveact", "epic_kitchens"]
STRIDES = [1, 2, 4, 8, 16]
N_CLASSES = {"ucf101": 101, "ssv2": 174, "hmdb51": 51, "diving48": 48,
             "finegym": 97, "autsl": 226, "driveact": 33, "epic_kitchens": 89}
SUFFIXES = ("", "_trainres224", "_res224")


def stride_curve(dash: pd.DataFrame, model: str, dataset: str) -> np.ndarray | None:
    """Top-1 (%) at coverage=100 for each stride, or None if unavailable."""
    sub = dash[(dash.model == model) & (dash.dataset == dataset)]
    if sub.empty:
        for suffix in SUFFIXES:
            f = SWEEP / f"{model}_{dataset}{suffix}" / "sweep_summary.csv"
            if f.exists():
                try:
                    sub = pd.read_csv(f)
                    break
                except Exception:
                    pass
    if sub is None or len(sub) == 0:
        return None
    sub = sub[sub.coverage == 100]
    try:
        return np.array([float(sub[sub.stride == s].top1.iloc[0]) * 100
                         for s in STRIDES])
    except Exception:
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dash = pd.read_csv(DASH)

    curves: dict[tuple[str, str], np.ndarray] = {}
    for m in MODELS:
        for d in DATASETS:
            c = stride_curve(dash, m, d)
            if c is not None:
                curves[(m, d)] = c

    # log2(stride) spacing: strides are geometric, so linear spacing would give
    # the 8->16 step the same weight as 1->2 and overweight the sparse end.
    x = np.log2(STRIDES)
    width = x[-1] - x[0]

    rows = []
    for d in DATASETS:
        per_model = {k: [] for k in
                     ("published", "unclipped", "auc", "relative", "normalized")}
        for m in MODELS:
            c = curves.get((m, d))
            if c is None:
                continue
            drop = c[0] - c[-1]
            per_model["published"].append(max(drop, 0.0))
            per_model["unclipped"].append(drop)
            # Mean gap below the dense-sampling level across the whole curve.
            trapezoid = getattr(np, "trapezoid", np.trapz)  # renamed in numpy 2
            per_model["auc"].append(float(trapezoid(c[0] - c, x) / width))
            per_model["relative"].append(100 * drop / c[0] if c[0] > 0 else np.nan)
            chance = 100.0 / N_CLASSES[d]
            head = c[0] - chance
            per_model["normalized"].append(100 * drop / head if head > 0 else np.nan)
        if not per_model["published"]:
            continue
        rows.append(dict(dataset=d, n_models=len(per_model["published"]),
                         **{k: float(np.nanmean(v)) for k, v in per_model.items()}))

    df = pd.DataFrame(rows).sort_values("published", ascending=False)
    df.to_csv(OUT / "tds_metric_variants.csv", index=False)

    print("=" * 78)
    print("Dataset demand under each TDS variant (higher = more demanding)")
    print("=" * 78)
    print(df.round(2).to_string(index=False))

    print("\n" + "=" * 78)
    print("Spearman of each variant against the published TDS ranking")
    print("=" * 78)
    for col in ("unclipped", "auc", "relative", "normalized"):
        ok = df[["published", col]].dropna()
        r, p = spearmanr(ok.published, ok[col])
        print(f"  {col:<12s} rho={r:+.3f}  p={p:.5f}  (n={len(ok)})")

    print("\nRank order under each variant:")
    for col in ("published", "unclipped", "auc", "relative", "normalized"):
        order = df.sort_values(col, ascending=False).dataset.tolist()
        print(f"  {col:<12s} {' > '.join(order)}")

    print(f"\nWrote {OUT / 'tds_metric_variants.csv'}")


if __name__ == "__main__":
    main()
