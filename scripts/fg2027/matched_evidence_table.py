"""Matched-evidence statistics quoted in the FG 2027 paper and supplementary.

Reads the per-(model, dataset, k) accuracies written by
scripts/accv2026/rebuttal_analyze_matched.py and restricts them to the datasets
the paper fine-tunes on. Kinetics-400 is left out: it is evaluated with
pretrained rather than fine-tuned backbones and is not one of the paper's
datasets. FineGym has no matched sweep.

Usage:  .venv/bin/python scripts/fg2027/matched_evidence_table.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
LONG = ROOT / "evaluations/accv2026/rebuttal/matched_evidence_long.csv"

# Mean stride 1->16 drop per architecture (main paper, full temporal table).
TABLE_DROP = {"timesformer": 10.29, "videomamba": 11.23, "mc3_18": 22.60,
              "r3d_18": 29.72, "r2plus1d_18": 30.03, "vivit": 30.05,
              "videomae": 32.77, "slowfast_r50": 42.11}
BUDGET = {"timesformer": 8, "videomamba": 8, "mc3_18": 16, "r3d_18": 16,
          "r2plus1d_18": 16, "videomae": 16, "vivit": 32, "slowfast_r50": 32}
KS = [1, 2, 4, 8]   # strictly matched: the smallest frame budget is 8


def main() -> None:
    df = pd.read_csv(LONG)
    df = df[(df.dataset != "kinetics400") & df.k.isin(KS)]
    print(f"datasets: {sorted(df.dataset.unique())}")
    piv = df.pivot_table(index="model", columns="k", values="top1", aggfunc="mean")
    piv["rel_loss_8to2_pct"] = 100 * (piv[8] - piv[2]) / piv[8]
    piv["B"] = piv.index.map(BUDGET)
    piv["table_drop"] = piv.index.map(TABLE_DROP)
    print(piv.sort_values("table_drop").round(1).to_string())

    idx = list(piv.index)
    drop = [TABLE_DROP[i] for i in idx]
    bud = [BUDGET[i] for i in idx]
    print(f"\nframe budget vs table drop: rho={spearmanr(bud, drop)[0]:+.3f} "
          f"p={spearmanr(bud, drop)[1]:.4f}")
    for k in KS:
        r, p = spearmanr(piv[k], [-d for d in drop])
        rb, pb = spearmanr(piv[k], bud)
        print(f"accuracy at k={k}: vs table ordering rho={r:+.3f} p={p:.4f} | "
              f"vs budget rho={rb:+.3f} p={pb:.3f} | best={piv[k].idxmax()} "
              f"worst={piv[k].idxmin()}")
    loss = piv[8] - piv[2]
    r, p = spearmanr(loss, bud)
    print(f"loss k=8->2 vs budget: rho={r:+.3f} p={p:.3f}")
    r, p = spearmanr(loss, drop)
    print(f"loss k=8->2 vs table drop: rho={r:+.3f} p={p:.3f}")
    print(f"relative loss range: {piv.rel_loss_8to2_pct.min():.0f}%"
          f"-{piv.rel_loss_8to2_pct.max():.0f}%")


if __name__ == "__main__":
    main()
