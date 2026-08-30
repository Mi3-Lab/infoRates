#!/usr/bin/env python3
"""Repeated-measures ANOVA over the coverage x stride grid (reviewer pxtb).

pxtb's statistical objection:

    "The coverage and stride measurements are repeated evaluations of the same
     clips, yet the paper applies standard ANOVA, Levene tests, and Welch tests
     without clearly defining independent sampling units or modeling repeated
     observations. The interaction between stride and coverage is also not
     reported."

The first half is correct. The published ANOVA treats the 25 cells as
independent, but every cell scores the *same* clips, so the residual is
contaminated by between-clip variance and the F ratios are not interpretable as
stated.

The second half is only partly right: an interaction *is* reported, in
supplementary S13, as a grid decomposition for two datasets (FineGym 15.0% of
variance, UCF-101 4.0%). It is not in the ANOVA model and does not cover all 64
pairs. This script supplies both.

Design: clip = subject, coverage (5) and stride (5) = fully crossed
within-subject factors. Each effect is tested against its own subject-by-factor
error term, and reported as partial eta squared

    partial_eta2 = SS_effect / (SS_effect + SS_error_effect)

which is the correct effect size for a within-subjects design and is directly
comparable to the eta^2 values in main paper Table 4.

Only clips present in all 25 cells are used, so the design stays balanced.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_repeated_measures_anova.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "evaluations/accv2026/coverage_stride_sweep"
OUT = ROOT / "evaluations/accv2026/rebuttal"

MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48",
            "finegym", "autsl", "driveact", "epic_kitchens"]
COVERAGES = [10, 25, 50, 75, 100]
STRIDES = [1, 2, 4, 8, 16]
SUFFIXES = ("", "_trainres224", "_res224")


def load_cube(model: str, dataset: str) -> np.ndarray | None:
    """Return an (n_clips, 5, 5) array of correctness, balanced across cells."""
    frames: dict[tuple[int, int], pd.Series] = {}
    for ci, cov in enumerate(COVERAGES):
        for si, stride in enumerate(STRIDES):
            got = None
            for suffix in SUFFIXES:
                f = SWEEP / f"{model}_{dataset}{suffix}" / f"cov{cov}_s{stride}_samples.csv"
                if f.exists():
                    try:
                        got = pd.read_csv(f)
                        break
                    except Exception:
                        pass
            if got is None:
                return None
            got = got[got["error"].isna() & ~got["skipped"].astype(bool)]
            frames[(ci, si)] = got.set_index("video_id")["correct_top1"].astype(float)

    shared = None
    for s in frames.values():
        idx = s.index[~s.index.duplicated()]
        shared = idx if shared is None else shared.intersection(idx)
    if shared is None or len(shared) < 30:
        return None

    cube = np.empty((len(shared), len(COVERAGES), len(STRIDES)), dtype=float)
    for (ci, si), s in frames.items():
        cube[:, ci, si] = s[~s.index.duplicated()].reindex(shared).to_numpy()
    return cube if np.isfinite(cube).all() else None


def rm_anova(cube: np.ndarray) -> dict:
    """Two-way within-subject ANOVA; each effect tested on its own error term."""
    n, a, b = cube.shape
    gm = cube.mean()

    m_sub = cube.mean(axis=(1, 2))          # subject means
    m_a = cube.mean(axis=(0, 2))            # coverage means
    m_b = cube.mean(axis=(0, 1))            # stride means
    m_ab = cube.mean(axis=0)                # coverage x stride cell means
    m_sa = cube.mean(axis=2)                # subject x coverage
    m_sb = cube.mean(axis=1)                # subject x stride

    ss_a = n * b * ((m_a - gm) ** 2).sum()
    ss_b = n * a * ((m_b - gm) ** 2).sum()
    ss_ab = n * ((m_ab - m_a[:, None] - m_b[None, :] + gm) ** 2).sum()

    # Error terms are the subject-by-effect interactions.
    ss_sa = b * ((m_sa - m_sub[:, None] - m_a[None, :] + gm) ** 2).sum()
    ss_sb = a * ((m_sb - m_sub[:, None] - m_b[None, :] + gm) ** 2).sum()
    resid = (cube
             - m_sa[:, :, None] - m_sb[:, None, :] - m_ab[None, :, :]
             + m_sub[:, None, None] + m_a[None, :, None] + m_b[None, None, :]
             - gm)
    ss_sab = (resid ** 2).sum()

    def peta2(ss_eff: float, ss_err: float) -> float:
        total = ss_eff + ss_err
        return float(ss_eff / total) if total > 0 else float("nan")

    df_a, df_b = a - 1, b - 1
    return dict(
        n_clips=n,
        peta2_coverage=peta2(ss_a, ss_sa),
        peta2_stride=peta2(ss_b, ss_sb),
        peta2_interaction=peta2(ss_ab, ss_sab),
        F_coverage=float((ss_a / df_a) / (ss_sa / (df_a * (n - 1)))) if ss_sa > 0 else np.nan,
        F_stride=float((ss_b / df_b) / (ss_sb / (df_b * (n - 1)))) if ss_sb > 0 else np.nan,
        F_interaction=float((ss_ab / (df_a * df_b))
                            / (ss_sab / (df_a * df_b * (n - 1)))) if ss_sab > 0 else np.nan,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            cube = load_cube(model, dataset)
            if cube is None:
                continue
            rows.append(dict(model=model, dataset=dataset, **rm_anova(cube)))
            r = rows[-1]
            print(f"{model:>13s}/{dataset:<14s} n={r['n_clips']:5d}  "
                  f"cov={r['peta2_coverage']:.3f}  stride={r['peta2_stride']:.3f}  "
                  f"cov*stride={r['peta2_interaction']:.3f}")

    if not rows:
        print("no usable pairs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "repeated_measures_anova.csv", index=False)

    print("\n" + "=" * 78)
    print(f"Partial eta^2, mean +/- sd over {len(df)} model-dataset pairs")
    print("=" * 78)
    for col, label in [("peta2_coverage", "coverage"),
                       ("peta2_stride", "stride"),
                       ("peta2_interaction", "coverage x stride")]:
        print(f"  {label:<18s} {df[col].mean():.3f} +/- {df[col].std():.3f}")

    print("\nPer-model stride effect (compare with main paper Table 4 eta^2_stride):")
    per_model = df.groupby("model")[["peta2_coverage", "peta2_stride",
                                     "peta2_interaction"]].mean()
    print(per_model.round(3).sort_values("peta2_stride").to_string())

    big = df.nlargest(5, "peta2_interaction")[
        ["model", "dataset", "peta2_interaction"]]
    print("\nLargest coverage x stride interactions:")
    print(big.round(3).to_string(index=False))
    print(f"\nWrote {OUT / 'repeated_measures_anova.csv'}")


if __name__ == "__main__":
    main()
