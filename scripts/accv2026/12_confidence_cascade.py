#!/usr/bin/env python3
"""Confidence-Based Cascade Routing for ACCV 2026.

Core idea: instead of running every video at a fixed budget k, run a cheap
k_low-frame inference first. If the model is confident (score > τ), stop.
Otherwise, upgrade to k_high frames. This is model-aware (semantic signal)
rather than FDE-based (kinematic signal).

Cascade levels:  4f → 8f → 16f → 32f
At each level, the routing decision uses the model's own softmax confidence.

Key properties:
- No new training required — simulated from existing samples CSVs
- Works for ANY model (uses per-video confidence scores)
- Directly comparable to fixed-budget baselines at equal average frames
- "Replace" semantics: if a video is upgraded from 4f to 16f, we run 16f
  (not 4f+16f), so avg_frames stays between k_low and k_high

Outputs per eval_dir:
  confidence_cascade_results.csv   — accuracy vs avg_frames for all τ values
  confidence_cascade_summary.csv   — best threshold per cascade pair
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGETS = [4, 8, 16, 32]


# ─── Single-level cascade ─────────────────────────────────────────────────────

def simulate_cascade_1level(
    df: pd.DataFrame,
    k_low: int,
    k_high: int,
    tau: float,
) -> dict:
    """Simulate a two-level cascade: run k_low first, upgrade to k_high if
    confidence < tau.

    'Replace' semantics: upgraded videos use only k_high frames (not k_low+k_high).
    This is fair: avg_frames is between k_low and k_high.

    Returns accuracy, avg_frames, fraction_upgraded.
    """
    low = df[df["budget"] == k_low][["video_id", "correct_top1", "confidence"]].copy()
    high = df[df["budget"] == k_high][["video_id", "correct_top1"]].copy()
    high.columns = ["video_id", "correct_top1_high"]

    merged = low.merge(high, on="video_id")
    if merged.empty:
        return {"accuracy": np.nan, "avg_frames": np.nan, "fraction_upgraded": np.nan}

    # Routing decision
    stay = merged["confidence"] >= tau
    merged["final_correct"] = np.where(stay, merged["correct_top1"], merged["correct_top1_high"])
    merged["final_frames"]  = np.where(stay, k_low, k_high)

    return {
        "accuracy":          float(merged["final_correct"].mean()),
        "avg_frames":        float(merged["final_frames"].mean()),
        "fraction_upgraded": float((~stay).mean()),
    }


# ─── Multi-level cascade ──────────────────────────────────────────────────────

def simulate_cascade_multilevel(
    df: pd.DataFrame,
    thresholds: list[float],  # [τ_4→8, τ_8→16, τ_16→32]
    budgets: list[int] = BUDGETS,
) -> dict:
    """Simulate a full 4→8→16→32 cascade with per-level thresholds.

    For each video:
      start at 4f; if confidence < τ1 → upgrade to 8f
      at 8f; if confidence < τ2 → upgrade to 16f
      at 16f; if confidence < τ3 → upgrade to 32f
    """
    assert len(thresholds) == len(budgets) - 1

    # Build per-video lookup: {video_id: {budget: (correct, confidence)}}
    pivot = {}
    for _, row in df[~df["skipped"].astype(bool)].iterrows():
        vid = str(row["video_id"])
        b   = int(row["budget"])
        if vid not in pivot:
            pivot[vid] = {}
        pivot[vid][b] = (bool(row["correct_top1"]), float(row["confidence"]))

    results = []
    for vid, bdata in pivot.items():
        final_budget = budgets[0]
        for i, b in enumerate(budgets[:-1]):
            if b not in bdata:
                break
            conf = bdata[b][1]
            if conf >= thresholds[i]:
                final_budget = b
                break
            final_budget = budgets[i + 1]

        correct = bdata.get(final_budget, (False, 0.0))[0] if final_budget in bdata else False
        results.append({"video_id": vid, "final_budget": final_budget, "correct": correct})

    rdf = pd.DataFrame(results)
    if rdf.empty:
        return {"accuracy": np.nan, "avg_frames": np.nan}

    return {
        "accuracy":   float(rdf["correct"].mean()),
        "avg_frames": float(rdf["final_budget"].mean()),
        "budget_dist": rdf["final_budget"].value_counts().sort_index().to_dict(),
    }


# ─── Threshold sweep ──────────────────────────────────────────────────────────

def sweep_cascade(
    df: pd.DataFrame,
    k_low: int,
    k_high: int,
    n_tau: int = 50,
) -> pd.DataFrame:
    """Sweep τ from 0→1 for a two-level cascade, return accuracy vs avg_frames."""
    taus = np.linspace(0.0, 1.0, n_tau)
    rows = []
    for tau in taus:
        res = simulate_cascade_1level(df, k_low, k_high, tau)
        rows.append({"tau": tau, "k_low": k_low, "k_high": k_high, **res})
    return pd.DataFrame(rows)


def pareto_gain(sweep_df: pd.DataFrame, fixed_results: dict) -> pd.DataFrame:
    """For each point on the cascade curve, compute delta_acc vs nearest fixed."""
    rows = []
    for _, r in sweep_df.iterrows():
        af = r["avg_frames"]
        # Nearest fixed budget
        nearest = min(BUDGETS, key=lambda b: abs(b - af))
        fixed_acc = fixed_results.get(nearest, np.nan)
        rows.append({
            "tau":          r["tau"],
            "avg_frames":   af,
            "accuracy":     r["accuracy"],
            "fixed_acc":    fixed_acc,
            "delta_acc":    r["accuracy"] - fixed_acc,
        })
    return pd.DataFrame(rows)


# ─── Per-dataset evaluation ───────────────────────────────────────────────────

def evaluate_all_cascades(samples_csv: Path, out_dir: Path) -> dict:
    """Run all cascade configurations for a given samples CSV.

    Returns summary dict with best result per cascade pair.
    """
    df = pd.read_csv(samples_csv)
    df = df[~df["skipped"].astype(bool)].copy()
    df["video_id"]    = df["video_id"].astype(str)
    df["correct_top1"] = df["correct_top1"].astype(bool)
    df["confidence"]   = df["confidence"].astype(float)

    model_name  = out_dir.name
    dataset     = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    n_videos    = df["video_id"].nunique()

    # Fixed baselines
    fixed = {}
    for b in BUDGETS:
        sub = df[df["budget"] == b]
        if len(sub) > 0:
            fixed[b] = float(sub["correct_top1"].mean())

    print(f"\n  {model_name} [{dataset}, n={n_videos}]")
    print(f"  Fixed: " + " | ".join(f"{b}f={v*100:.1f}%" for b, v in sorted(fixed.items())))

    all_sweeps = []
    summary_rows = []

    # Two-level cascades: all adjacent pairs
    cascade_pairs = [(4, 8), (4, 16), (4, 32), (8, 16), (8, 32), (16, 32)]
    for k_low, k_high in cascade_pairs:
        if k_low not in fixed or k_high not in fixed:
            continue
        sweep = sweep_cascade(df, k_low, k_high, n_tau=100)
        sweep["model"]   = model_name
        sweep["dataset"] = dataset
        gain = pareto_gain(sweep, fixed)
        sweep["delta_acc"] = gain["delta_acc"].values
        all_sweeps.append(sweep)

        # Best: max accuracy gain at avg_frames < k_high
        sub = sweep[sweep["avg_frames"] < k_high - 0.5]
        if sub.empty:
            continue
        best = sub.loc[sub["delta_acc"].idxmax()]
        summary_rows.append({
            "model":       model_name,
            "dataset":     dataset,
            "k_low":       k_low,
            "k_high":      k_high,
            "best_tau":    round(float(best["tau"]), 3),
            "accuracy":    round(float(best["accuracy"]), 4),
            "avg_frames":  round(float(best["avg_frames"]), 2),
            "delta_acc":   round(float(best["delta_acc"]), 4),
            "fixed_low":   round(fixed[k_low], 4),
            "fixed_high":  round(fixed[k_high], 4),
        })
        sign = "+" if best["delta_acc"] > 0 else ""
        print(f"  Cascade {k_low}f→{k_high}f: best tau={best['tau']:.2f} → "
              f"acc={best['accuracy']*100:.2f}% @ {best['avg_frames']:.1f}f avg "
              f"({sign}{best['delta_acc']*100:.2f}% vs fixed)")

    # Multi-level cascade: optimize over 3 thresholds jointly
    best_ml_acc, best_ml_res, best_ml_taus = 0.0, {}, [0.5, 0.5, 0.5]
    target_avg = 12  # frames — midpoint between 8 and 16
    best_ml_delta = -np.inf
    # Coarse grid search over 3 thresholds (5×5×5 = 125 points)
    for t1, t2, t3 in product(np.linspace(0.1, 0.9, 5),
                               np.linspace(0.1, 0.9, 5),
                               np.linspace(0.1, 0.9, 5)):
        res = simulate_cascade_multilevel(df, [t1, t2, t3])
        if np.isnan(res["accuracy"]):
            continue
        nearest = min(fixed, key=lambda b: abs(b - res["avg_frames"]))
        delta = res["accuracy"] - fixed[nearest]
        if delta > best_ml_delta:
            best_ml_delta = delta
            best_ml_res   = res
            best_ml_taus  = [t1, t2, t3]

    if best_ml_res:
        nearest = min(fixed, key=lambda b: abs(b - best_ml_res["avg_frames"]))
        summary_rows.append({
            "model":       model_name,
            "dataset":     dataset,
            "k_low":       "4",
            "k_high":      "32",
            "best_tau":    str([round(t, 2) for t in best_ml_taus]),
            "accuracy":    round(best_ml_res["accuracy"], 4),
            "avg_frames":  round(best_ml_res["avg_frames"], 2),
            "delta_acc":   round(best_ml_delta, 4),
            "fixed_low":   round(fixed.get(4, np.nan), 4),
            "fixed_high":  round(fixed.get(32, np.nan), 4),
        })
        sign = "+" if best_ml_delta > 0 else ""
        print(f"  Multi-level 4→8→16→32: taus={[round(t,2) for t in best_ml_taus]} → "
              f"acc={best_ml_res['accuracy']*100:.2f}% @ {best_ml_res['avg_frames']:.1f}f avg "
              f"({sign}{best_ml_delta*100:.2f}% vs Fixed-{nearest}f)")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_sweeps:
        pd.concat(all_sweeps, ignore_index=True).to_csv(
            out_dir / "confidence_cascade_sweep.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "confidence_cascade_summary.csv", index=False)

    return {"fixed": fixed, "summary": summary_df, "model": model_name, "dataset": dataset}


# ─── Figure ───────────────────────────────────────────────────────────────────

def plot_cascade_curves(all_results: list[dict], out_path: Path):
    """Plot accuracy vs avg_frames for cascade and fixed baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by dataset
    datasets = list({r["dataset"] for r in all_results})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]

    for ax, dataset in zip(axes, sorted(datasets)):
        ds_results = [r for r in all_results if r["dataset"] == dataset]

        # Fixed baselines (dashed gray)
        for r in ds_results[:1]:
            for b, acc in sorted(r["fixed"].items()):
                ax.axhline(acc * 100, color="gray", linestyle="--", alpha=0.4, linewidth=1)
                ax.annotate(f"Fixed-{b}f\n{acc*100:.1f}%", xy=(b, acc * 100),
                            fontsize=7, color="gray", ha="center", va="bottom")

        # Cascade curves — best 4→16 and multilevel per model
        for i, r in enumerate(ds_results):
            if r["summary"].empty:
                continue
            # 4→16 cascade
            sw_path = Path("evaluations/accv2026/confidence_cascade") / r["model"] / "confidence_cascade_sweep.csv"
            if sw_path.exists():
                sw = pd.read_csv(sw_path)
                pair = sw[(sw["k_low"] == 4) & (sw["k_high"] == 16)]
                if not pair.empty:
                    pair = pair.sort_values("avg_frames")
                    ax.plot(pair["avg_frames"], pair["accuracy"] * 100,
                            color=colors[i % len(colors)], linewidth=1.5,
                            label=r["model"], alpha=0.8)

        ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Avg frames used")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(3, 33)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Confidence Cascade: Accuracy vs Avg Frames\n(vs fixed-budget baselines, dashed)",
                 fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

# Map eval_dir → samples CSV pattern for all known datasets
EVAL_CONFIGS = [
    # SSV2
    ("r3d18_ssv2_full_e10_a100",        "somethingv2_*_samples.csv",       "R3D-18"),
    ("mc3_18_ssv2_full_e10_a100",        "somethingv2_*_samples.csv",       "MC3-18"),
    ("r2plus1d_18_ssv2_full_e10_a100",   "somethingv2_*_samples.csv",       "R2plus1D-18"),
    ("timesformer_ssv2_full_e10_h200",   "somethingv2_*_samples.csv",       "TimeSformer"),
    ("vivit_ssv2_full_e10_h200",         "somethingv2_*_samples.csv",       "ViViT"),
    ("slowfast_r50_ssv2_full_e10_a100",  "somethingv2_*_samples.csv",       "SlowFast"),
    ("videomae_ssv2_full_e5_h200",       "somethingv2_*_samples.csv",       "VideoMAE"),
    # UCF101
    ("r2plus1d_18_ucf101_full_e10_a100", "ucf101_*_samples.csv",            "R2plus1D-18"),
    ("slowfast_r50_ucf101_full_e10_a100","ucf101_*_samples.csv",            "SlowFast"),
    ("videomae_ucf101_full_e10_h200",    "ucf101_*_samples.csv",            "VideoMAE"),
    # HMDB51
    ("r2plus1d_18_hmdb51_full_e10_a100", "hmdb51_*_samples.csv",            "R2plus1D-18"),
    ("slowfast_r50_hmdb51_full_e10_a100","hmdb51_*_samples.csv",            "SlowFast"),
    ("videomae_hmdb51_full_e10_h200",    "hmdb51_*_samples.csv",            "VideoMAE"),
    # Diving48
    ("r2plus1d_18_diving48_full_e10_a100","diving48_*_samples.csv",         "R2plus1D-18"),
    ("slowfast_r50_diving48_full_e10_a100","diving48_*_samples.csv",        "SlowFast"),
    ("videomae_diving48_full_e10_h200",   "diving48_*_samples.csv",         "VideoMAE"),
    # EPIC-Kitchens
    ("r2plus1d_18_epic_kitchens_full_e10_a100", "epic_kitchens_*_samples.csv", "R2plus1D-18"),
    ("videomae_epic_kitchens_full_e10_h200",    "epic_kitchens_*_samples.csv", "VideoMAE"),
    # WLASL100 — replaced by AUTSL
    # AUTSL (Turkish Sign Language) — added when training completes
    # ("r2plus1d_18_autsl_full_e10_a100",       "autsl_*_samples.csv",         "R2plus1D-18"),
    # ("videomae_autsl_full_e10_h200",          "autsl_*_samples.csv",         "VideoMAE"),
    # Drive&Act — added when training completes
    # ("r2plus1d_18_driveact_full_e10_a100",    "driveact_*_samples.csv",      "R2plus1D-18"),
    # ("videomae_driveact_full_e10_h200",       "driveact_*_samples.csv",      "VideoMAE"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-base", default="evaluations/accv2026/fixed_budget",
                        help="Base dir containing eval subdirectories")
    parser.add_argument("--output-dir", default="evaluations/accv2026/confidence_cascade",
                        help="Output directory for cascade results")
    parser.add_argument("--datasets", nargs="+",
                        default=["ssv2", "ucf101", "hmdb51", "diving48"],
                        help="Which datasets to run (ssv2 ucf101 hmdb51 diving48)")
    args = parser.parse_args()

    eval_base = ROOT / args.eval_base
    out_base  = ROOT / args.output_dir
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CONFIDENCE CASCADE ROUTING — ACCV 2026")
    print("=" * 65)

    all_results  = []
    all_summaries = []

    for eval_dir, pattern, model_label in EVAL_CONFIGS:
        # Filter by requested datasets
        ds_match = any(ds in eval_dir for ds in args.datasets)
        if not ds_match:
            continue

        eval_path = eval_base / eval_dir
        if not eval_path.exists():
            continue

        matches = list(eval_path.glob(pattern))
        if not matches:
            continue
        samples_csv = matches[0]

        out_dir = out_base / eval_dir
        res = evaluate_all_cascades(samples_csv, out_dir)
        res["model"] = model_label
        all_results.append(res)
        if not res["summary"].empty:
            res["summary"]["model"] = model_label
            all_summaries.append(res["summary"])

    if not all_summaries:
        print("\nNo results found.")
        return

    # Global summary
    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all.to_csv(out_base / "cascade_global_summary.csv", index=False)

    # Best gains table
    print("\n" + "=" * 65)
    print("TOP CASCADE GAINS (delta_acc > 0, sorted by delta):")
    print("=" * 65)
    gains = summary_all[summary_all["delta_acc"] > 0].sort_values("delta_acc", ascending=False)
    if not gains.empty:
        print(gains[["model", "dataset", "k_low", "k_high",
                      "accuracy", "avg_frames", "delta_acc", "best_tau"]
              ].head(20).to_string(index=False))
    else:
        print("No positive gains found.")

    # Figure
    fig_path = out_base / "fig_confidence_cascade.pdf"
    try:
        plot_cascade_curves(all_results, fig_path)
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")

    print(f"\nAll results saved to {out_base}/")
    print(f"Global summary: {out_base}/cascade_global_summary.csv")


if __name__ == "__main__":
    main()
