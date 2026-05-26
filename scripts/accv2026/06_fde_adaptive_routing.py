#!/usr/bin/env python3
"""Frame Difference Energy (FDE) analysis and adaptive routing for ACCV 2026.

Workflow:
1. Load per-video fixed-budget results (samples CSV)
2. Determine optimal budget per video = min budget where correct
3. Compute FDE for each video (probe signal using 4 low-res frames)
4. Measure FDE vs optimal budget correlation
5. Optimize FDE thresholds → adaptive router
6. Report: adaptive accuracy vs average frame count vs fixed budgets
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


# ─── FDE computation ───────────────────────────────────────────────────────────

def compute_fde(video_path: str, n_probe: int = 5, size: int = 64) -> Optional[float]:
    """Sample n_probe frames, return mean absolute frame difference (normalized 0-1).

    Uses PyAV to be consistent with the training data pipeline.
    Returns None if the video is unreadable.
    """
    try:
        import av
    except ImportError:
        raise RuntimeError("PyAV not installed: pip install av")

    try:
        container = av.open(video_path, timeout=10)
        stream = container.streams.video[0]
        total = stream.frames or 0
        duration = float(stream.duration or 0) * stream.time_base if stream.duration else 0

        frames_out = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            arr = frame.to_ndarray(format="gray")  # H×W uint8
            # Resize to size×size using simple slice
            h, w = arr.shape
            arr = arr[::max(1, h // size), ::max(1, w // size)][:size, :size].astype(np.float32)
            frames_out.append(arr)
            if len(frames_out) >= n_probe * 4:  # oversample then stride
                break
        container.close()

        if len(frames_out) < 2:
            return None

        # Stride to n_probe evenly spaced
        indices = np.linspace(0, len(frames_out) - 1, n_probe, dtype=int)
        frames_out = [frames_out[i] for i in indices]

        diffs = [np.mean(np.abs(frames_out[i + 1].astype(float) - frames_out[i].astype(float)))
                 for i in range(len(frames_out) - 1)]
        return float(np.mean(diffs)) / 255.0

    except Exception:
        return None


# ─── Optimal budget per video ─────────────────────────────────────────────────

def compute_optimal_budgets(samples_df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    """Return DataFrame with video_id, optimal_budget, never_correct.

    optimal_budget = min budget where correct_top1=True; if never correct → max budget.
    """
    rows = []
    for vid_id, grp in samples_df.groupby("video_id"):
        valid = grp[~grp["skipped"].astype(bool) & grp["correct_top1"].astype(bool)]
        if valid.empty:
            never = True
            opt = max(budgets)
        else:
            never = False
            opt = int(valid["budget"].min())
        rows.append({"video_id": vid_id, "optimal_budget": opt, "never_correct": never,
                     "video_path": grp["video_path"].iloc[0]})
    return pd.DataFrame(rows)


# ─── Router evaluation ────────────────────────────────────────────────────────

def evaluate_router(
    samples_df: pd.DataFrame,
    fde_df: pd.DataFrame,
    thresholds: list[float],
    budgets: list[int],
) -> dict:
    """Given FDE thresholds, assign each video a budget and compute accuracy.

    thresholds has len(budgets)-1 values: budget[i] used when fde < thresholds[i].
    """
    merged = samples_df.merge(fde_df[["video_id", "fde"]], on="video_id", how="inner")
    merged = merged[~merged["skipped"].astype(bool)]

    def assign_budget(fde: float) -> int:
        for thresh, budget in zip(thresholds, budgets[:-1]):
            if fde < thresh:
                return budget
        return budgets[-1]

    merged["assigned_budget"] = merged["fde"].apply(assign_budget)
    # Keep only the row matching the assigned budget
    matched = merged[merged["budget"] == merged["assigned_budget"]]

    n_correct = matched["correct_top1"].sum()
    n_total = matched["video_id"].nunique()
    avg_frames = matched["assigned_budget"].mean()
    accuracy = n_correct / max(1, len(matched))

    return {
        "accuracy": float(accuracy),
        "avg_frames": float(avg_frames),
        "n_videos": int(n_total),
        "n_matched": int(len(matched)),
    }


def fixed_budget_accuracy(samples_df: pd.DataFrame, budget: int) -> dict:
    sub = samples_df[(samples_df["budget"] == budget) & ~samples_df["skipped"].astype(bool)]
    return {
        "accuracy": float(sub["correct_top1"].mean()),
        "avg_frames": float(budget),
        "n_videos": int(sub["video_id"].nunique()),
    }


# ─── Threshold optimization ───────────────────────────────────────────────────

def optimize_thresholds(
    samples_df: pd.DataFrame,
    fde_df: pd.DataFrame,
    budgets: list[int],
    n_grid: int = 20,
) -> tuple[list[float], dict]:
    """Grid-search FDE thresholds to maximize accuracy at fixed avg_frames target."""
    merged = samples_df.merge(fde_df[["video_id", "fde"]], on="video_id", how="inner")
    merged = merged[~merged["skipped"].astype(bool)]

    fde_vals = fde_df["fde"].dropna()
    t_range = np.linspace(fde_vals.quantile(0.05), fde_vals.quantile(0.95), n_grid)

    # For 4 budgets, we need 3 thresholds
    # Simple sweep for 2-budget case (lower/upper threshold)
    best_result = None
    best_score = -1.0
    best_thresholds: list[float] = []

    # For each pair of thresholds (t1 < t2 < t3)
    import itertools
    threshold_combos = list(itertools.combinations(t_range, len(budgets) - 1))

    for thresholds in threshold_combos:
        thresholds = list(sorted(thresholds))
        result = evaluate_router(samples_df, fde_df, thresholds, budgets)
        # Score: accuracy - 0.01 * avg_frames (penalize cost)
        score = result["accuracy"] - 0.005 * result["avg_frames"]
        if score > best_score:
            best_score = score
            best_result = result
            best_thresholds = thresholds

    return best_thresholds, best_result or {}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-csv", required=True, help="Fixed-budget samples CSV")
    p.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--n-probe", type=int, default=5, help="Frames to sample for FDE")
    p.add_argument("--fde-cache", default=None, help="Path to cache FDE values CSV")
    p.add_argument("--output-dir", default=None, help="Output dir for results")
    p.add_argument("--max-videos", type=int, default=0, help="Limit for quick testing")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    samples_csv = Path(args.samples_csv)
    out_dir = Path(args.output_dir) if args.output_dir else samples_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples: {samples_csv}")
    df = pd.read_csv(samples_csv)
    print(f"  {len(df)} rows, {df['video_id'].nunique()} unique videos")

    opt_df = compute_optimal_budgets(df, args.budgets)
    print(f"\nOptimal budget distribution:")
    print(opt_df["optimal_budget"].value_counts().sort_index().to_string())
    print(f"Never correct: {opt_df['never_correct'].sum()}")

    # FDE computation (with caching)
    fde_cache_path = Path(args.fde_cache) if args.fde_cache else out_dir / "fde_cache.csv"
    if fde_cache_path.exists():
        print(f"\nLoading cached FDE values: {fde_cache_path}")
        fde_df = pd.read_csv(fde_cache_path)
    else:
        print(f"\nComputing FDE for {len(opt_df)} videos (workers={args.workers})...")
        video_paths = opt_df[["video_id", "video_path"]].drop_duplicates()
        if args.max_videos > 0:
            video_paths = video_paths.iloc[:args.max_videos]

        import concurrent.futures
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(compute_fde, row["video_path"], args.n_probe): row["video_id"]
                for _, row in video_paths.iterrows()
            }
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                vid_id = futures[fut]
                fde = fut.result()
                results.append({"video_id": vid_id, "fde": fde})
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(video_paths)} done")

        fde_df = pd.DataFrame(results)
        fde_df.to_csv(fde_cache_path, index=False)
        print(f"  Saved FDE cache: {fde_cache_path}")

    fde_df = fde_df.dropna(subset=["fde"])
    print(f"\nFDE stats (n={len(fde_df)}):")
    print(f"  mean={fde_df['fde'].mean():.4f}  median={fde_df['fde'].median():.4f}"
          f"  std={fde_df['fde'].std():.4f}  min={fde_df['fde'].min():.4f}  max={fde_df['fde'].max():.4f}")

    # Merge FDE with optimal budget for correlation analysis
    merged = opt_df.merge(fde_df, on="video_id", how="inner")
    print(f"\nFDE by optimal budget:")
    print(merged.groupby("optimal_budget")["fde"].agg(["mean", "median", "std"]).round(4).to_string())

    # Spearman correlation: FDE vs optimal budget
    from scipy import stats
    corr, pval = stats.spearmanr(merged["fde"], merged["optimal_budget"])
    print(f"\nSpearman r(FDE, optimal_budget) = {corr:.4f}  p={pval:.4e}")

    # Fixed-budget baselines
    print("\nFixed-budget baselines:")
    fixed_results = {}
    for b in args.budgets:
        r = fixed_budget_accuracy(df, b)
        fixed_results[b] = r
        print(f"  Fixed-{b:2d}f: acc={r['accuracy']:.4f}  avg_frames={r['avg_frames']:.1f}")

    # Optimize adaptive router
    print("\nOptimizing FDE thresholds...")
    best_thresholds, best_result = optimize_thresholds(df, fde_df, args.budgets, n_grid=15)
    print(f"  Best thresholds: {[f'{t:.4f}' for t in best_thresholds]}")
    print(f"  Adaptive: acc={best_result.get('accuracy',0):.4f}  avg_frames={best_result.get('avg_frames',0):.2f}")

    # Compare with fixed-budget at similar frame count
    target_frames = best_result.get("avg_frames", 16)
    print(f"\nComparison at ~{target_frames:.1f} avg frames:")
    print(f"  Adaptive router: acc={best_result.get('accuracy',0):.4f}")
    for b, r in fixed_results.items():
        print(f"  Fixed-{b:2d}f:       acc={r['accuracy']:.4f}")

    # Save results
    results_rows = []
    for b, r in fixed_results.items():
        results_rows.append({"method": f"fixed_{b}f", "accuracy": r["accuracy"],
                              "avg_frames": r["avg_frames"], "type": "fixed"})
    results_rows.append({"method": "fde_adaptive", "accuracy": best_result.get("accuracy", 0),
                          "avg_frames": best_result.get("avg_frames", 0),
                          "thresholds": str(best_thresholds), "type": "adaptive"})
    results_df = pd.DataFrame(results_rows)
    out_path = out_dir / "fde_routing_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results: {out_path}")

    # Save FDE-budget correlation table
    corr_table = merged.groupby("optimal_budget")["fde"].agg(["mean", "median", "std", "count"]).round(4)
    corr_table.to_csv(out_dir / "fde_budget_correlation.csv")
    print(f"Saved correlation table: {out_dir}/fde_budget_correlation.csv")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Spearman r(FDE, optimal_budget) = {corr:.4f}  (p={pval:.2e})")
    print(f"  Adaptive (FDE router): acc={best_result.get('accuracy',0)*100:.1f}%"
          f"  avg={best_result.get('avg_frames',0):.1f}f")
    print(f"  Fixed-16f:             acc={fixed_results.get(16,{}).get('accuracy',0)*100:.1f}%"
          f"  avg=16.0f")
    gain = (best_result.get('accuracy', 0) - fixed_results.get(16, {}).get('accuracy', 0)) * 100
    frame_savings = 16 - best_result.get('avg_frames', 16)
    print(f"  Gain vs Fixed-16f: {gain:+.1f}% acc, {frame_savings:+.1f}f avg savings")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
