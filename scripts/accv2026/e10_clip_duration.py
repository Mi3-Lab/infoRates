"""E10 — Clip Duration Analysis.

Tests whether clip duration (short vs long videos) affects temporal aliasing
sensitivity. Uses existing E1 samples which record source_frames per video.

For each model/dataset, groups videos by duration and measures:
- Accuracy at stride=1 (dense) vs stride=16 (sparse)
- Aliasing sensitivity per duration bin

Outputs:
  evaluations/accv2026/e10_duration/{model}_{dataset}_duration.csv
  evaluations/accv2026/e10_duration/duration_summary.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("evaluations/accv2026/coverage_stride_sweep")
OUT  = Path("evaluations/accv2026/e10_duration")
OUT.mkdir(parents=True, exist_ok=True)

MODELS   = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
            "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens"]

# Approximate FPS per dataset (for duration estimation)
DATASET_FPS = {
    "ssv2": 12, "ucf101": 25, "hmdb51": 25,
    "diving48": 25, "autsl": 25, "driveact": 15, "epic_kitchens": 60,
}

# Duration bins in seconds
DURATION_BINS   = [0, 1, 3, 6, 1000]
DURATION_LABELS = ["<1s", "1-3s", "3-6s", ">6s"]

summary_rows = []

for model in MODELS:
    for dataset in DATASETS:
        fps = DATASET_FPS.get(dataset, 25)
        f_dense  = BASE / f"{model}_{dataset}" / "cov100_s1_samples.csv"
        f_sparse = BASE / f"{model}_{dataset}" / "cov100_s16_samples.csv"

        if not f_dense.exists() or not f_sparse.exists():
            continue

        def clean(df):
            df = df[df["error"].isna() & ~df["skipped"].astype(bool)].copy()
            df = df[df["source_frames"] > 0].copy()
            return df

        dense  = clean(pd.read_csv(f_dense))
        sparse = clean(pd.read_csv(f_sparse))

        if len(dense) < 50:
            continue

        # Merge on video_id
        merged = dense[["video_id","source_frames","correct_top1"]].merge(
            sparse[["video_id","correct_top1"]].rename(columns={"correct_top1":"correct_s16"}),
            on="video_id", how="inner"
        )
        if len(merged) < 50:
            continue

        # Compute duration
        merged["duration_s"] = merged["source_frames"] / fps
        merged["duration_bin"] = pd.cut(
            merged["duration_s"],
            bins=DURATION_BINS, labels=DURATION_LABELS
        )

        # Per-bin stats
        grp = merged.groupby("duration_bin", observed=True).agg(
            n=("correct_top1", "count"),
            acc_dense=("correct_top1", "mean"),
            acc_sparse=("correct_s16", "mean"),
        ).reset_index()
        grp["aliasing_loss_pp"] = (grp["acc_dense"] - grp["acc_sparse"]) * 100
        grp["model"]   = model
        grp["dataset"] = dataset

        # Only include bins with ≥20 videos
        grp = grp[grp["n"] >= 20]

        if len(grp) < 2:
            continue

        grp.to_csv(OUT / f"{model}_{dataset}_duration.csv", index=False)

        # Check correlation: longer clip → more aliasing?
        valid = grp.dropna(subset=["aliasing_loss_pp"])
        if len(valid) >= 3:
            from scipy import stats
            # Use midpoint of bin as numeric duration
            bin_mids = {"<1s": 0.5, "1-3s": 2.0, "3-6s": 4.5, ">6s": 8.0}
            valid["bin_mid"] = valid["duration_bin"].astype(str).map(bin_mids)
            r, p = stats.pearsonr(valid["bin_mid"], valid["aliasing_loss_pp"])

            for _, row in valid.iterrows():
                summary_rows.append({
                    "model": model, "dataset": dataset,
                    "duration_bin": str(row["duration_bin"]),
                    "n": row["n"],
                    "acc_dense": row["acc_dense"],
                    "acc_sparse": row["acc_sparse"],
                    "aliasing_loss_pp": row["aliasing_loss_pp"],
                    "pearson_r_dur_aliasing": r,
                    "pearson_p": p,
                })

            print(f"  {model}/{dataset}: {len(valid)} duration bins, "
                  f"r(duration, aliasing)={r:.3f} p={p:.3f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "duration_summary.csv", index=False)

# Print key findings
print("\n" + "="*75)
print("E10 — Clip Duration vs Aliasing: Key findings (TimeSformer)")
print("="*75)

tsf = summary_df[summary_df["model"] == "timesformer"]
if not tsf.empty:
    pivot = tsf.pivot_table(
        index="duration_bin", columns="dataset",
        values="aliasing_loss_pp", aggfunc="mean"
    )
    print(f"\nAliasing loss (pp) by clip duration — stride=1→16, cov=100%")
    print(pivot.round(1).to_string())

print("\n" + "="*75)
print("AGGREGATE: Pearson r(clip duration, aliasing loss) per model")
print("Positive r = longer clips → more aliasing")
print("="*75)

if not summary_df.empty:
    agg = summary_df.groupby(["model","dataset"])[["pearson_r_dur_aliasing","pearson_p"]].first()
    mean_r = summary_df.groupby("model")["pearson_r_dur_aliasing"].mean()
    print(f"\n{'Model':<16} {'mean r':>8} {'interpretation'}")
    print("-"*45)
    for model, r in mean_r.sort_values(ascending=False).items():
        interp = "longer → more aliasing" if r > 0.3 else \
                 "longer → less aliasing" if r < -0.3 else "no clear trend"
        print(f"{model:<16} {r:>8.3f}  {interp}")
