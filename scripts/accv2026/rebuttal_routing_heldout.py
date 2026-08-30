#!/usr/bin/env python3
"""Held-out evaluation of the confidence cascade (reviewer pxtb).

The submitted protocol sweeps tau on the validation split and reports the best
point from that same split, so the +0.2pp over fixed-dense inference cannot be
distinguished from threshold-selection noise. Here tau is chosen on a
calibration half and scored on the untouched other half, repeated over many
stratified splits to attach an interval to the gain.

Also note for the write-up: the rule is `max_c p(c|v) > tau`, i.e. a maximum
softmax probability threshold, not an entropy threshold. Reviewer pxtb is
correct that "entropy routing" is a misnomer.

Cheap config = cov100/s4 (4 effective frames); dense = cov100/s1 (16).

Usage:  .venv/bin/python scripts/accv2026/rebuttal_routing_heldout.py
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

THRESHOLDS = np.round(np.arange(0.10, 0.96, 0.05), 2)
CHEAP_FRAMES, DENSE_FRAMES = 4.0, 16.0
FRAME_BUDGET = 8.0        # same constraint as the submitted table
N_SPLITS = 1000
RNG = np.random.default_rng(0)


def load_pair(model: str, dataset: str) -> pd.DataFrame | None:
    """Per-clip cheap/dense outcomes joined on video_id."""
    frames = {}
    for tag, stride in (("cheap", 4), ("dense", 1)):
        got = None
        for suffix in ("", "_trainres224", "_res224"):
            f = SWEEP / f"{model}_{dataset}{suffix}" / f"cov100_s{stride}_samples.csv"
            if f.exists():
                try:
                    got = pd.read_csv(f)
                    break
                except Exception:
                    pass
        if got is None:
            return None
        got = got[got["error"].isna() & ~got["skipped"].astype(bool)]
        frames[tag] = got[["video_id", "label_id", "confidence", "correct_top1"]]
    merged = frames["cheap"].merge(
        frames["dense"], on=["video_id", "label_id"], suffixes=("_cheap", "_dense"))
    return merged if len(merged) >= 50 else None


def route(df: pd.DataFrame, tau: float) -> tuple[float, float]:
    """Accuracy and average frame cost when routing at threshold tau."""
    cheap = df.confidence_cheap.values > tau
    correct = np.where(cheap, df.correct_top1_cheap.values, df.correct_top1_dense.values)
    frames = np.where(cheap, CHEAP_FRAMES, DENSE_FRAMES)
    return float(correct.mean()), float(frames.mean())


def pick_tau(df: pd.DataFrame) -> float:
    """Best tau on this split subject to the average-frame budget."""
    best_tau, best_acc = THRESHOLDS[0], -1.0
    for tau in THRESHOLDS:
        acc, frames = route(df, tau)
        if frames <= FRAME_BUDGET and acc > best_acc:
            best_tau, best_acc = tau, acc
    return best_tau


def stratified_half(df: pd.DataFrame, rng) -> np.ndarray:
    """Boolean mask selecting ~half of each class for calibration."""
    mask = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("label_id").indices.items():
        idx = np.asarray(idx)
        rng.shuffle(idx)
        mask[idx[: max(1, len(idx) // 2)]] = True
    return mask


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in MODELS:
        for dataset in DATASETS:
            df = load_pair(model, dataset)
            if df is None:
                continue
            df = df.reset_index(drop=True)

            in_sample_tau = pick_tau(df)
            in_sample_acc, in_sample_frames = route(df, in_sample_tau)

            gains, accs, frames_used, taus = [], [], [], []
            for _ in range(N_SPLITS):
                calib = stratified_half(df, RNG)
                tr, te = df[calib], df[~calib]
                if len(te) < 20 or len(tr) < 20:
                    continue
                tau = pick_tau(tr)
                acc, frames = route(te, tau)
                accs.append(acc)
                frames_used.append(frames)
                taus.append(tau)
                gains.append(acc - te.correct_top1_dense.mean())

            if not accs:
                continue
            gains = np.array(gains) * 100
            rows.append(dict(
                model=model, dataset=dataset, n_clips=len(df),
                # what the submitted table reports
                in_sample_tau=in_sample_tau,
                in_sample_acc=round(100 * in_sample_acc, 2),
                in_sample_frames=round(in_sample_frames, 2),
                in_sample_gain_vs_dense=round(
                    100 * (in_sample_acc - df.correct_top1_dense.mean()), 2),
                # honest held-out version
                heldout_acc=round(100 * float(np.mean(accs)), 2),
                heldout_frames=round(float(np.mean(frames_used)), 2),
                heldout_gain_vs_dense=round(float(gains.mean()), 2),
                gain_ci_lo=round(float(np.percentile(gains, 2.5)), 2),
                gain_ci_hi=round(float(np.percentile(gains, 97.5)), 2),
                pct_splits_positive=round(100 * float((gains > 0).mean()), 1),
                tau_median=float(np.median(taus)),
            ))
            r = rows[-1]
            print(f"{model:>13s}/{dataset:<14s} "
                  f"in-sample {r['in_sample_gain_vs_dense']:+.2f}pp -> "
                  f"held-out {r['heldout_gain_vs_dense']:+.2f}pp "
                  f"[{r['gain_ci_lo']:+.2f},{r['gain_ci_hi']:+.2f}] "
                  f"({r['pct_splits_positive']:.0f}% splits > 0)")

    if not rows:
        print("no usable (model, dataset) pairs found")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "routing_heldout.csv", index=False)

    print("\n" + "=" * 78)
    print(f"Summary over {len(out)} model-dataset pairs "
          f"({N_SPLITS} stratified 50/50 splits each)")
    print("=" * 78)
    print(f"  mean in-sample gain vs fixed dense : "
          f"{out.in_sample_gain_vs_dense.mean():+.2f} pp")
    print(f"  mean held-out  gain vs fixed dense : "
          f"{out.heldout_gain_vs_dense.mean():+.2f} pp")
    print(f"  pairs whose held-out 95% CI excludes 0 : "
          f"{int(((out.gain_ci_lo > 0) | (out.gain_ci_hi < 0)).sum())}/{len(out)}")

    ssv2 = out[(out.model == 'timesformer') & (out.dataset == 'ssv2')]
    if not ssv2.empty:
        r = ssv2.iloc[0]
        print(f"\n  TimeSformer/SSv2 (the headline cell):")
        print(f"    submitted (in-sample) : {r.in_sample_acc:.1f}% @ "
              f"{r.in_sample_frames:.1f} frames, {r.in_sample_gain_vs_dense:+.2f}pp")
        print(f"    held-out              : {r.heldout_acc:.1f}% @ "
              f"{r.heldout_frames:.1f} frames, {r.heldout_gain_vs_dense:+.2f}pp "
              f"[{r.gain_ci_lo:+.2f}, {r.gain_ci_hi:+.2f}]")
    print(f"\nWrote {OUT / 'routing_heldout.csv'}")


if __name__ == "__main__":
    main()
