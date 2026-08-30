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
    c_reported = c[c.resolution.isin([48, 96, 112, 160, 224])]
    chk("convergence logs at five reported resolutions = 503", 503,
        len(c_reported), 0)
    lo = c[c.resolution == 48]
    chk("48px still improving 52.8%", 52.8,
        round(100 * lo.still_improving.mean(), 1), 0.1)
    chk("48px last-3-epoch gain 1.38pp", 1.38, round(lo.gain_last_3.mean(), 2), 0.02)

    # Matched evidence: the two claims the architecture argument rests on.
    from scipy.stats import spearmanr
    PUBLISHED_DROP = {"timesformer": 10.29, "videomamba": 11.23, "mc3_18": 22.60,
                      "r3d_18": 29.72, "r2plus1d_18": 30.03, "vivit": 30.05,
                      "videomae": 32.77, "slowfast_r50": 42.11}
    BUDGET = {"timesformer": 8, "videomamba": 8, "r3d_18": 16, "mc3_18": 16,
              "r2plus1d_18": 16, "videomae": 16, "vivit": 32, "slowfast_r50": 32}
    m = pd.read_csv(R / "matched_evidence_long.csv")
    m = m[m.strictly_matched & (m.dataset != "kinetics400")]
    full = m.model.nunique()
    have = m.groupby("dataset").model.nunique()
    m = m[m.dataset.isin(have[have == full].index)]
    piv = m.pivot_table(index="model", columns="k", values="top1", aggfunc="mean")
    idx = [i for i in piv.index if i in PUBLISHED_DROP]
    r_acc, _ = spearmanr([piv.loc[i, 2] for i in idx],
                         [-PUBLISHED_DROP[i] for i in idx])
    chk("matched: accuracy at k=2 vs published rho=0.857", 0.857,
        round(float(r_acc), 3), 0.005)
    drop = piv[8] - piv[2]
    r_bud, _ = spearmanr([drop[i] for i in idx], [BUDGET[i] for i in idx])
    chk("matched: budget confound collapses to rho=0.231", 0.231,
        round(float(r_bud), 3), 0.005)

    # Span: the sampling-rate axis, and the count of context-bound datasets.
    sp = pd.read_csv(R / "span_long.csv")
    n_ctx = 0
    for _, g in sp.groupby("dataset"):
        mm = g.groupby("span_pct").top1.mean()
        if len(mm) >= 3 and spearmanr(mm.index, mm.values)[0] > 0.5:
            n_ctx += 1
    chk("span: context-bound datasets = 7", 7, n_ctx, 0)

    # TRA: the recovery figures quoted for both architectures.
    tra = pd.read_csv(R / "tra_ablation.csv")
    for model, expected in (("timesformer", 28.4), ("r3d_18", 37.0)):
        row = tra[tra.model == model]
        if not row.empty:
            chk(f"TRA {model} paper arm recovers {expected}%", expected,
                float(row.paper_recovered_pct.iloc[0]), 0.15)

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
