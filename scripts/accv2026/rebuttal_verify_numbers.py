#!/usr/bin/env python3
"""Check every number quoted in the rebuttal against the CSV that produced it.

In a rebuttal the cost of a wrong number is not symmetric with the cost of a
missing one: a reviewer who catches one discrepancy will discount the rest of
the response, including the parts that are correct. Run this before submitting,
and again after any edit to paper/rebuttal.tex or the analysis scripts.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_verify_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "evaluations/accv2026/rebuttal"

checks: list[tuple[str, float, float | None, bool]] = []


def chk(label: str, claimed: float, actual: float | None, tol: float = 0.05) -> None:
    checks.append((label, claimed, actual,
                   actual is not None and abs(claimed - actual) <= tol))


def main() -> int:
    d = pd.read_csv(R / "routing_heldout.csv")
    row = d[(d.model == "timesformer") & (d.dataset == "ssv2")].iloc[0]
    chk("routing in-sample +0.21pp", 0.21, row.in_sample_gain_vs_dense)
    chk("routing held-out -0.03pp", -0.03, row.heldout_gain_vs_dense)
    chk("routing CI low -1.07", -1.07, row.gain_ci_lo)
    chk("routing CI high +0.83", 0.83, row.gain_ci_hi)
    chk("routing mean over 56 pairs -3.5pp", -3.5,
        round(d.heldout_gain_vs_dense.mean(), 2), 0.06)

    a = pd.read_csv(R / "repeated_measures_anova.csv")
    chk("partial eta2 coverage 0.292", 0.292, round(a.peta2_coverage.mean(), 3), 0.001)
    chk("partial eta2 stride 0.232", 0.232, round(a.peta2_stride.mean(), 3), 0.001)
    chk("partial eta2 interaction 0.070", 0.070,
        round(a.peta2_interaction.mean(), 3), 0.001)
    chk("ANOVA pairs = 56", 56, len(a), 0)

    p = pd.read_csv(R / "paired_drops.csv").dropna(subset=["drop_unpadded"])
    chk("drop, all clips +10.7pp", 10.74, round(p.drop_all.mean(), 2), 0.06)
    chk("drop, unpadded clips -0.5pp", -0.51, round(p.drop_unpadded.mean(), 2), 0.06)
    chk("pairs with unpadded subset = 21", 21, len(p), 0)

    c = pd.read_csv(R / "convergence_curves.csv")
    lo = c[c.resolution == 48]
    chk("48px still improving 52.8%", 52.8,
        round(100 * lo.still_improving.mean(), 1), 0.1)
    chk("48px last-3-epoch gain 1.38pp", 1.38, round(lo.gain_last_3.mean(), 2), 0.02)

    width = max(len(l) for l, *_ in checks)
    print(f"{'claim':<{width}}{'quoted':>10}{'actual':>10}")
    for label, claimed, actual, ok in checks:
        shown = f"{actual}" if actual is not None else "--"
        print(f"{label:<{width}}{claimed:>10}{shown:>10}   "
              f"{'ok' if ok else '*** MISMATCH ***'}")

    bad = [c for c in checks if not c[3]]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} verified"
          + (f" — {len(bad)} MISMATCHED" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
