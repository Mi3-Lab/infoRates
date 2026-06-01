"""E2 — Variance Analysis of temporal aliasing across action classes.

For each (model, dataset, coverage, stride) configuration, computes:
  - Per-class accuracy (correct_top1 rate per label_id)
  - Std of per-class accuracy across classes (inter-class variance)
  - Levene's test: does variance change as stride increases?

Outputs:
  evaluations/accv2026/e2_variance/{model}_{dataset}_perclass.csv
  evaluations/accv2026/e2_variance/levene_results.csv
  evaluations/accv2026/e2_variance/variance_summary.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import json, sys

BASE = Path("evaluations/accv2026/coverage_stride_sweep")
OUT  = Path("evaluations/accv2026/e2_variance")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens"]
COVERAGES = [10, 25, 50, 75, 100]
STRIDES   = [1, 2, 4, 8, 16]

levene_rows = []
var_rows    = []

for model in MODELS:
    for dataset in DATASETS:
        sweep_dir = BASE / f"{model}_{dataset}"
        if not sweep_dir.exists():
            continue

        # Load all sample CSVs for this model/dataset
        dfs = []
        for cov in COVERAGES:
            for s in STRIDES:
                f = sweep_dir / f"cov{cov}_s{s}_samples.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    df["coverage"] = cov
                    df["stride"]   = s
                    dfs.append(df)

        if not dfs:
            continue

        data = pd.concat(dfs, ignore_index=True)
        data = data[data["error"].isna() & ~data["skipped"].astype(bool)].copy()
        data["correct"] = data["correct_top1"].astype(bool).astype(int)

        print(f"  {model}/{dataset}: {len(data)} samples across {len(dfs)} configs")

        # Per-class accuracy at each (coverage, stride)
        perclass_rows = []
        for cov in COVERAGES:
            for s in STRIDES:
                sub = data[(data["coverage"] == cov) & (data["stride"] == s)]
                if sub.empty:
                    continue
                grp = sub.groupby("label_id")["correct"].agg(["sum","count"])
                grp["acc"] = grp["sum"] / grp["count"]
                grp["coverage"] = cov
                grp["stride"]   = s
                grp["model"]    = model
                grp["dataset"]  = dataset
                perclass_rows.append(grp.reset_index())

        if not perclass_rows:
            continue

        perclass = pd.concat(perclass_rows, ignore_index=True)
        perclass.to_csv(OUT / f"{model}_{dataset}_perclass.csv", index=False)

        # Variance of per-class accuracy at each (cov, stride)
        for cov in COVERAGES:
            for s in STRIDES:
                sub = perclass[(perclass["coverage"] == cov) & (perclass["stride"] == s)]
                if len(sub) < 3:
                    continue
                var_rows.append({
                    "model": model, "dataset": dataset,
                    "coverage": cov, "stride": s,
                    "n_classes": len(sub),
                    "mean_acc":  sub["acc"].mean(),
                    "std_acc":   sub["acc"].std(),
                    "min_acc":   sub["acc"].min(),
                    "max_acc":   sub["acc"].max(),
                })

        # Levene's test: does stride affect inter-class variance?
        # Compare distributions at stride=1 vs stride=16 (both at coverage=100)
        for cov in [100]:
            group_s1  = perclass[(perclass["coverage"]==cov) & (perclass["stride"]==1)]["acc"].dropna()
            group_s16 = perclass[(perclass["coverage"]==cov) & (perclass["stride"]==16)]["acc"].dropna()
            if len(group_s1) < 3 or len(group_s16) < 3:
                continue
            stat, pval = stats.levene(group_s1, group_s16)
            # Variance ratio: how much more spread at stride=16 vs stride=1?
            var_ratio = group_s16.std() / (group_s1.std() + 1e-9)
            levene_rows.append({
                "model": model, "dataset": dataset, "coverage": cov,
                "stride_a": 1, "stride_b": 16,
                "std_s1":   group_s1.std(),
                "std_s16":  group_s16.std(),
                "var_ratio_16_over_1": var_ratio,
                "levene_stat": stat, "levene_p": pval,
                "significant": pval < 0.05,
                "interpretation": "variance INCREASES with stride" if var_ratio > 1.2 else
                                  "variance stable" if var_ratio < 0.8 else "marginal change",
            })

# Save results
var_df = pd.DataFrame(var_rows)
var_df.to_csv(OUT / "variance_summary.csv", index=False)
print(f"\nVariance summary: {len(var_df)} rows → {OUT}/variance_summary.csv")

lev_df = pd.DataFrame(levene_rows)
lev_df.to_csv(OUT / "levene_results.csv", index=False)
print(f"Levene results:  {len(lev_df)} rows → {OUT}/levene_results.csv")

# Print key findings
print("\n" + "="*70)
print("E2 KEY FINDINGS — Variance at stride=1 vs stride=16 (cov=100%)")
print("="*70)
print(f"\n{'Model':<16} {'Dataset':<14} {'std@s1':>8} {'std@s16':>8} {'ratio':>7} {'p-val':>8} {'sig?':>5}")
print("-"*70)
for _, row in lev_df.sort_values("var_ratio_16_over_1", ascending=False).iterrows():
    sig = "✅" if row["significant"] else "  "
    print(f"{row['model']:<16} {row['dataset']:<14} "
          f"{row['std_s1']:>8.3f} {row['std_s16']:>8.3f} "
          f"{row['var_ratio_16_over_1']:>7.2f}x {row['levene_p']:>8.4f} {sig}")
