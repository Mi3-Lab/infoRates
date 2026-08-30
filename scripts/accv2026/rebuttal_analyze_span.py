#!/usr/bin/env python3
"""The sampling-rate axis the submitted sweep never varied (reviewer vdwp).

vdwp's objection is that the temporal protocol measures "how early the model has
enough temporal context", not aliasing. The `span` sweep addresses it directly:
the frame count k is pinned to each model's native budget and only the width W
of the (centred) window varies, so the realised sampling rate k/W varies by an
order of magnitude while the number of frames delivered never changes.

Reading this requires care, and the care is the point. At fixed k, widening W
does two things at once:

  * it lowers the sampling rate    (fewer samples per second of action)
  * it widens the temporal context (more of the action falls inside the window)

These push in opposite directions, which is exactly why the two cannot be
separated by frame subsampling alone -- the identifiability limitation stated in
the revised introduction. What the sweep can still decide is which dominates:

  accuracy rises with W    context is binding; the model is starved of action
                           extent, not of sampling density
  accuracy falls with W    density is binding; coarser sampling of the same
                           action costs more than the added context gains
  accuracy flat in W       neither binds over the tested range

Only the antialias sweep can attribute a decline specifically to aliasing, since
it fixes both k and W and varies only spectral content. This script establishes
which regime we are in.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_analyze_span.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "evaluations/accv2026/rebuttal_sweeps/span"
OUT = ROOT / "evaluations/accv2026/rebuttal"
BUDGET = {"timesformer": 8, "videomamba": 8, "r3d_18": 16, "mc3_18": 16,
          "r2plus1d_18": 16, "videomae": 16, "vivit": 32, "slowfast_r50": 32}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SWEEPS.exists():
        print(f"no span results yet under {SWEEPS}")
        return

    rows = []
    for f in sorted(SWEEPS.glob("*/summary.csv")):
        d = pd.read_csv(f)
        model, dataset = d.model.iloc[0], d.dataset.iloc[0]
        for r in d.itertuples():
            rows.append(dict(model=model, dataset=dataset,
                             span_pct=int(str(r.config).replace("span", "")),
                             top1=100 * r.top1, distinct=r.mean_distinct))
    if not rows:
        print("no span configurations parsed")
        return

    df = pd.DataFrame(rows)
    df["budget"] = df.model.map(BUDGET)
    df.to_csv(OUT / "span_long.csv", index=False)

    print(f"{df.groupby(['model','dataset']).ngroups} (model, dataset) pairs, "
          f"{df.dataset.nunique()} datasets\n")

    print("=" * 78)
    print("Accuracy (%) vs window width, k pinned to each model's frame budget")
    print("=" * 78)
    print(df.pivot_table(index="dataset", columns="span_pct", values="top1",
                         aggfunc="mean").round(2).to_string())

    print("\nBy architecture:")
    print(df.pivot_table(index="model", columns="span_pct", values="top1",
                         aggfunc="mean").round(2).to_string())

    print("\n" + "=" * 78)
    print("Which constraint binds, per dataset?")
    print("=" * 78)
    print(f"{'dataset':<15s}{'rho(W, acc)':>13s}{'p':>9s}{'10->100%':>11s}   regime")
    verdicts = []
    for dataset, g in df.groupby("dataset"):
        m = g.groupby("span_pct").top1.mean()
        if len(m) < 3:
            continue
        r, p = spearmanr(m.index, m.values)
        delta = m.get(100, float("nan")) - m.get(10, float("nan"))
        if r > 0.5:
            regime = "context-bound (wider window helps)"
        elif r < -0.5:
            regime = "density-bound (coarser sampling hurts)"
        else:
            regime = "neither binds clearly"
        verdicts.append(regime)
        print(f"{dataset:<15s}{r:13.3f}{p:9.4f}{delta:+11.1f}   {regime}")

    if verdicts:
        n_ctx = sum("context" in v for v in verdicts)
        print(f"\n  context-bound: {n_ctx}/{len(verdicts)} datasets")
        print("  A context-bound result supports vdwp directly: over the range we")
        print("  can test, the sweep is limited by how much of the action is seen,")
        print("  not by how densely it is sampled. Attributing any density effect")
        print("  to aliasing requires the antialias sweep, which fixes both k and")
        print("  W and varies only the spectral content.")

    print("\n  Sanity check -- distinct frames delivered should be constant in W:")
    print(df.pivot_table(index="model", columns="span_pct", values="distinct",
                         aggfunc="mean").round(1).to_string())

    print(f"\nWrote {OUT / 'span_long.csv'}")


if __name__ == "__main__":
    main()
