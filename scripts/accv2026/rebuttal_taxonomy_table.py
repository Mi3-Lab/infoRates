#!/usr/bin/env python3
"""Regenerate the action-sensitivity taxonomy table (reviewer vdwp).

vdwp: "Table 5 doesn't have FineGym."

True, and there is a second problem underneath it: the published thresholds are
not reproducible from the current artifacts. Recomputing tertiles of per-class
aliasing loss over the full 8-model pool gives, for AUTSL, Low 46.7 / High 57.3,
while the paper prints 66.9 / 73.9; SSv2 gives 20.6 / 31.8 against a printed
59.7 / 76.9. The class counts match because a tertile split always yields n/3
per tier, so they are not evidence of agreement. The printed values look like
they came from a smaller, more fragile model pool -- CNN-only reproduces that
direction -- but nothing in the repo pins it down.

So patching one FineGym row into the published table would print a row computed
one way beside seven computed another. This regenerates all eight rows from the
same pool and emits the LaTeX.

Per-class aliasing loss is the accuracy drop from stride 1 to 16 at coverage
100%, averaged over every model with data for that class, exactly as
e5_taxonomy_analysis.py defines it.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_taxonomy_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAX = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT = ROOT / "evaluations/accv2026/rebuttal"

DISPLAY = {"autsl": "AUTSL", "finegym": "FineGym", "ssv2": "SSv2",
           "driveact": "DriveAct", "diving48": "Diving-48", "hmdb51": "HMDB-51",
           "epic_kitchens": "EPIC-Kitchens", "ucf101": "UCF-101"}
# What main paper Table 5 currently prints, for the delta column.
PUBLISHED = {"autsl": (66.9, 73.9), "diving48": (25.0, 47.7), "ssv2": (59.7, 76.9),
             "hmdb51": (8.4, 28.5), "epic_kitchens": (0.0, 19.4),
             "driveact": (14.8, 51.0), "ucf101": (1.4, 7.5)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, label in DISPLAY.items():
        f = TAX / f"{dataset}_class_taxonomy.csv"
        if not f.exists():
            print(f"[skip] no taxonomy file for {dataset}")
            continue
        df = pd.read_csv(f)
        drop = df.mean_abs_drop * 100
        lo, hi = drop.quantile(0.33), drop.quantile(0.67)
        pub = PUBLISHED.get(dataset)
        rows.append(dict(
            dataset=label, classes=len(df),
            low=round(float(lo), 1), high=round(float(hi), 1),
            n_high=int((drop >= hi).sum()), n_low=int((drop < lo).sum()),
            n_models=int(df.n_models.median()),
            published_low=pub[0] if pub else None,
            published_high=pub[1] if pub else None,
        ))

    if not rows:
        print("no taxonomy data found")
        return
    tab = pd.DataFrame(rows).sort_values("high", ascending=False)
    tab.to_csv(OUT / "taxonomy_table_regenerated.csv", index=False)

    print("=" * 84)
    print("Regenerated taxonomy tiers (all 8 datasets, full model pool)")
    print("=" * 84)
    print(tab.to_string(index=False))

    print("\nDivergence from the printed Table 5:")
    for r in tab.itertuples():
        if r.published_low is None:
            print(f"  {r.dataset:<15s} not in the printed table (this is vdwp's point)")
            continue
        print(f"  {r.dataset:<15s} low {r.low:6.1f} vs {r.published_low:6.1f} "
              f"({r.low - r.published_low:+6.1f})   "
              f"high {r.high:6.1f} vs {r.published_high:6.1f} "
              f"({r.high - r.published_high:+6.1f})")

    print("\n" + "=" * 84)
    print("LaTeX (drop-in replacement for Table 5)")
    print("=" * 84)
    print(r"\begin{tabular}{l r r r r}")
    print(r"\toprule")
    print(r"Dataset & Low $\leq$ & High $>$ & \# High & \# Low \\")
    print(r"\midrule")
    for r in tab.itertuples():
        print(f"{r.dataset:<15s} & {r.low:5.1f}pp & {r.high:5.1f}pp "
              f"& {r.n_high:3d} & {r.n_low:3d} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\nWrote {OUT / 'taxonomy_table_regenerated.csv'}")


if __name__ == "__main__":
    main()
