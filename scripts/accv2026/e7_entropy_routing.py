"""E7 — Entropy/Confidence-based Adaptive Routing.

Uses E1 sweep samples (already collected) to simulate routing without any new
GPU inference. For each video, uses the model's confidence at a cheap config
(few frames, high stride) to decide whether to upgrade to dense inference.

Routing logic:
  cheap config  = (cov=25%, stride=4)  → ~1 effective frame, fast
  dense config  = (cov=100%, stride=1) → 16 frames, full quality

  if confidence(cheap) > τ: use cheap result  (model is sure with few frames)
  else:                      use dense result  (model is uncertain, needs more)

This directly uses the aliasing curves from E1 to motivate routing decisions.

Outputs:
  evaluations/accv2026/e7_routing/{model}_{dataset}_routing.csv
  evaluations/accv2026/e7_routing/e7_summary.csv
  evaluations/accv2026/e7_routing/e7_vs_baselines.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("evaluations/accv2026/coverage_stride_sweep")
OUT  = Path("evaluations/accv2026/e7_routing")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens"]

# Routing configs: cheap proxy → dense target
# Use cov100%/stride=4 (4 actual frames) as cheap — directly comparable to FrameExit(4→16)
# Use cov100%/stride=1 (16 frames) as dense
CHEAP_COV, CHEAP_STRIDE = 100, 4   # 4 actual frames (like FrameExit stage 1)
DENSE_COV,  DENSE_STRIDE = 100, 1  # 16 actual frames (like FrameExit stage 2)

# Additional budget levels for multi-level routing
BUDGET_CONFIGS = [
    (10, 16, "~0.1f"),   # ultra cheap
    (25,  4, "~1f"),     # cheap (primary proxy)
    (50,  2, "~4f"),     # medium
    (100, 1, "~16f"),    # full (dense)
]

THRESHOLDS = np.arange(0.1, 1.0, 0.05)

all_summary_rows = []
baseline_rows    = []

for model in MODELS:
    for dataset in DATASETS:
        sweep_dir = BASE / f"{model}_{dataset}"

        # Load cheap and dense sample CSVs
        f_cheap = sweep_dir / f"cov{CHEAP_COV}_s{CHEAP_STRIDE}_samples.csv"
        f_dense = sweep_dir / f"cov{DENSE_COV}_s{DENSE_STRIDE}_samples.csv"

        if not f_cheap.exists() or not f_dense.exists():
            continue

        cheap = pd.read_csv(f_cheap)
        dense = pd.read_csv(f_dense)

        # Filter valid samples
        def clean(df):
            return df[df["error"].isna() & ~df["skipped"].astype(bool)].copy()

        cheap = clean(cheap)
        dense = clean(dense)

        # Merge on video_id
        merged = cheap[["video_id","confidence","correct_top1"]].merge(
            dense[["video_id","correct_top1"]].rename(columns={"correct_top1":"correct_dense"}),
            on="video_id", how="inner"
        )

        if len(merged) < 100:
            continue

        # Skip feature collapse models
        cheap_acc = merged["correct_top1"].mean()
        dense_acc = merged["correct_dense"].mean()
        if cheap_acc < 0.05 and dense_acc < 0.05:
            print(f"  SKIP {model}/{dataset}: feature collapse")
            continue

        # Frames used by each config
        frames_cheap = 16 * CHEAP_COV / 100 / CHEAP_STRIDE   # ≈ 1 frame
        frames_dense = 16 * DENSE_COV / 100 / DENSE_STRIDE    # = 16 frames

        # Fixed baselines
        baseline_rows.append({"model": model, "dataset": dataset,
                               "method": f"fixed_cheap(cov{CHEAP_COV}_s{CHEAP_STRIDE})",
                               "accuracy": cheap_acc, "avg_frames": frames_cheap})
        baseline_rows.append({"model": model, "dataset": dataset,
                               "method": f"fixed_dense(cov{DENSE_COV}_s{DENSE_STRIDE})",
                               "accuracy": dense_acc, "avg_frames": frames_dense})

        # Oracle: for each video, pick whichever config is correct
        oracle_correct = (merged["correct_top1"] | merged["correct_dense"]).mean()

        # Threshold sweep
        routing_rows = []
        for tau in THRESHOLDS:
            # Route: if confidence > tau → use cheap, else → use dense
            use_cheap = merged["confidence"] > tau
            correct = np.where(use_cheap, merged["correct_top1"], merged["correct_dense"])
            avg_frames = np.where(use_cheap, frames_cheap, frames_dense).mean()

            routing_rows.append({
                "model": model, "dataset": dataset,
                "threshold": round(tau, 2),
                "accuracy": correct.mean(),
                "avg_frames": avg_frames,
                "pct_cheap": use_cheap.mean(),
                "pct_dense": (1 - use_cheap.mean()),
                "oracle_accuracy": oracle_correct,
                "fixed_cheap_acc": cheap_acc,
                "fixed_dense_acc": dense_acc,
                "delta_vs_cheap": correct.mean() - cheap_acc,
                "delta_vs_dense": correct.mean() - dense_acc,
            })

        routing_df = pd.DataFrame(routing_rows)
        routing_df.to_csv(OUT / f"{model}_{dataset}_routing.csv", index=False)

        # Find best threshold: max accuracy at avg_frames < 8 (budget constraint)
        feasible = routing_df[routing_df["avg_frames"] <= 8]
        if not feasible.empty:
            best_idx = feasible["accuracy"].idxmax()
            best = feasible.loc[best_idx].to_dict()
            all_summary_rows.append({
                "model": model, "dataset": dataset,
                "best_tau":        best["threshold"],
                "best_accuracy":   best["accuracy"],
                "best_avg_frames": best["avg_frames"],
                "pct_cheap":       best["pct_cheap"],
                "fixed_cheap_acc": cheap_acc,
                "fixed_dense_acc": dense_acc,
                "oracle_accuracy": oracle_correct,
                "gain_vs_cheap":   best["accuracy"] - cheap_acc,
                "gain_vs_dense":   best["accuracy"] - dense_acc,
                "oracle_gap":      oracle_correct - best["accuracy"],
            })
            gain_cheap = best['accuracy'] - cheap_acc
            gain_dense = best['accuracy'] - dense_acc
            print(f"  {model}/{dataset}: τ={best['threshold']:.2f} → "
                  f"acc={best['accuracy']:.3f} ({gain_cheap:+.3f} vs cheap, "
                  f"{gain_dense:+.3f} vs dense) "
                  f"avg={best['avg_frames']:.1f}f "
                  f"oracle_gap={oracle_correct - best['accuracy']:.3f}")

# Save
summary_df = pd.DataFrame(all_summary_rows)
summary_df.to_csv(OUT / "e7_summary.csv", index=False)

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(OUT / "e7_baselines.csv", index=False)

# Print aggregate results per model
print("\n" + "="*80)
print("E7 — Entropy Routing: Aggregate results (avg across datasets, budget ≤8f)")
print("="*80)
print(f"\n{'Model':<16} {'n':>3} | {'E7 acc':>8} {'Δ_cheap':>9} {'Δ_dense':>9} | "
      f"{'oracle':>8} {'gap':>7} | {'%cheap':>7}")
print("-"*75)

for model, grp in summary_df.groupby("model"):
    print(f"{model:<16} {len(grp):>3} | "
          f"{grp['best_accuracy'].mean():>8.3f} "
          f"{grp['gain_vs_cheap'].mean():>+9.3f} "
          f"{grp['gain_vs_dense'].mean():>+9.3f} | "
          f"{grp['oracle_accuracy'].mean():>8.3f} "
          f"{grp['oracle_gap'].mean():>+7.3f} | "
          f"{grp['pct_cheap'].mean():>6.1%}")

print()
print("Δ_cheap = gain vs always-cheap; Δ_dense = gain vs always-dense (negative=worse)")
print("%cheap  = fraction of videos routed to cheap inference (savings)")
