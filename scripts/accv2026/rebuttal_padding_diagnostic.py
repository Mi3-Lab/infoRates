"""Diagnostic for the ACCV'26 rebuttal: what does the `stride` axis actually vary?

Reviewers pxtb and vdwp argue the temporal sweep does not measure aliasing.
This script tests that claim directly against the per-clip sweep outputs.

Three findings, each written to evaluations/accv2026/rebuttal/:

  1. index_demo      -- select_frame_indices() re-uniformizes the candidate pool
                        via linspace, so at fixed coverage the selected frames
                        span the same window at every stride. Stride only bites
                        when ceil(window/s) < budget, and the deficit is filled
                        by repeating the LAST candidate frame.
  2. padding_rates   -- fraction of clips in that padding regime, per
                        (model, dataset, stride).
  3. paired_drops    -- stride 1->16 accuracy drop on all clips vs. on clips
                        that never pad, paired on video_id.

Usage:  .venv/bin/python scripts/accv2026/rebuttal_padding_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from info_rates.evaluation.benchmark import select_frame_indices  # noqa: E402

SWEEP = ROOT / "evaluations/accv2026/coverage_stride_sweep"
OUT = ROOT / "evaluations/accv2026/rebuttal"

# Frame budgets as configured in sweep_coverage_stride.py.
BUDGET = {
    "timesformer": 8, "videomamba": 8,
    "r3d_18": 16, "mc3_18": 16, "r2plus1d_18": 16, "videomae": 16,
    "vivit": 32, "slowfast_r50": 32,
}
# videomamba has no per-clip samples files, only aggregate sweep_summary.csv.
MODELS = [m for m in BUDGET if m != "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48",
            "finegym", "autsl", "driveact", "epic_kitchens"]
# Mean stride 1->16 drop at cov=100%, reproduced from the sweep (paper Table 3).
AVG_DROP = {
    "timesformer": 10.29, "videomamba": 11.23, "mc3_18": 22.60, "r3d_18": 29.72,
    "r2plus1d_18": 30.03, "vivit": 30.05, "videomae": 32.77, "slowfast_r50": 42.11,
}
# Native-resolution results live in the bare dir; a few pairs only exist under
# an explicit 224px tag, so fall back the same way generate_paper_figures.py does.
SUFFIXES = ["", "_trainres224", "_res224"]


def load(model: str, dataset: str, cov: int, stride: int) -> pd.DataFrame | None:
    for suffix in SUFFIXES:
        f = SWEEP / f"{model}_{dataset}{suffix}" / f"cov{cov}_s{stride}_samples.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        return df[~df.skipped.astype(bool)]
    return None


def index_demo() -> pd.DataFrame:
    """Show that stride does not change the sampled span until padding starts."""
    rows = []
    for total in (400, 60):
        for stride in (1, 2, 4, 8, 16):
            idx = select_frame_indices(total, 16, 100, stride)
            rows.append(dict(
                total_frames=total, budget=16, stride=stride,
                distinct=len(set(idx.tolist())),
                span_first=int(idx[0]), span_last=int(idx[-1]),
                mean_gap=round(float(np.diff(idx).mean()), 2),
                indices=" ".join(map(str, idx.tolist())),
            ))
    return pd.DataFrame(rows)


def padding_rates() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for stride in (1, 2, 4, 8, 16):
                df = load(model, dataset, 100, stride)
                if df is None or df.empty:
                    continue
                padded = df.candidate_frames < df.model_input_frames
                rows.append(dict(
                    dataset=dataset, model=model, budget=BUDGET[model],
                    stride=stride, n_clips=len(df),
                    pct_padded=round(100 * float(padded.mean()), 1),
                    mean_distinct=round(float(
                        df.candidate_frames.clip(upper=df.model_input_frames).mean()), 2),
                ))
    return pd.DataFrame(rows)


def paired_drops() -> pd.DataFrame:
    """Drop from stride 1 to 16 on all clips vs. clips that never pad."""
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            dense, sparse = load(model, dataset, 100, 1), load(model, dataset, 100, 16)
            if dense is None or sparse is None:
                continue
            dense = dense.set_index("video_id")
            sparse = sparse.set_index("video_id")
            shared = dense.index.intersection(sparse.index)
            if len(shared) < 20:
                continue
            dense, sparse = dense.loc[shared], sparse.loc[shared]
            clean = (sparse.candidate_frames >= sparse.model_input_frames).values
            row = dict(
                dataset=dataset, model=model, budget=BUDGET[model],
                n_all=len(shared),
                drop_all=round(100 * float(
                    dense.correct_top1.mean() - sparse.correct_top1.mean()), 2),
                n_unpadded=int(clean.sum()),
                pct_unpadded=round(100 * float(clean.mean()), 1),
                drop_unpadded=np.nan,
            )
            if clean.sum() >= 20:
                row["drop_unpadded"] = round(100 * float(
                    dense.correct_top1[clean].mean() - sparse.correct_top1[clean].mean()), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    demo, pad, paired = index_demo(), padding_rates(), paired_drops()
    demo.to_csv(OUT / "index_demo.csv", index=False)
    pad.to_csv(OUT / "padding_rates.csv", index=False)
    paired.to_csv(OUT / "paired_drops.csv", index=False)

    print("=" * 78)
    print("1. Selected indices at cov=100%, budget=16")
    print("=" * 78)
    for total, grp in demo.groupby("total_frames", sort=False):
        print(f"\n  clip of T={total} frames")
        for r in grp.itertuples():
            print(f"    s={r.stride:<3d} distinct={r.distinct:<3d} "
                  f"span=[{r.span_first},{r.span_last}] mean_gap={r.mean_gap}")

    print("\n" + "=" * 78)
    print("2. Padding rate at stride 16, cov=100% (% of clips)")
    print("=" * 78)
    piv = (pad[pad.stride == 16]
           .pivot_table(index="dataset", columns="budget", values="pct_padded"))
    print(piv.round(1).to_string())

    print("\n" + "=" * 78)
    print("3. Stride 1->16 drop: all clips vs. clips that never pad")
    print("=" * 78)
    usable = paired.dropna(subset=["drop_unpadded"])
    print(paired.to_string(index=False))
    if not usable.empty:
        print(f"\n  pairs with an unpadded subset of n>=20: {len(usable)}")
        print(f"  mean drop, all clips      : {usable.drop_all.mean():+.2f} pp")
        print(f"  mean drop, unpadded clips : {usable.drop_unpadded.mean():+.2f} pp")
        print("  CAVEAT: unpadded clips are the longest ones, so this subset is")
        print("  length-confounded; read it alongside findings 1 and 4.")

    print("\n" + "=" * 78)
    print("4. Frame budget vs. reported temporal robustness")
    print("=" * 78)
    models = sorted(BUDGET, key=lambda m: AVG_DROP[m])
    rho, p = spearmanr([BUDGET[m] for m in models], [AVG_DROP[m] for m in models])
    for m in models:
        print(f"  {m:<14s} budget={BUDGET[m]:<3d} avg drop={AVG_DROP[m]:5.2f} pp")
    print(f"\n  Spearman(budget, avg drop) = {rho:.3f}  (p = {p:.4f}, n = {len(models)})")
    print(f"\nWrote index_demo.csv, padding_rates.csv, paired_drops.csv to {OUT}")


if __name__ == "__main__":
    main()
