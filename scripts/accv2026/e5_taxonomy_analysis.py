"""E5 — Action Sensitivity Taxonomy.

Classifies each action class into High/Moderate/Low aliasing sensitivity
using the per-class accuracy data from E1 samples.

Sensitivity = relative accuracy drop from stride=1 to stride=16 (cov=100%).

Outputs:
  evaluations/accv2026/e5_taxonomy/{dataset}_class_taxonomy.csv
  evaluations/accv2026/e5_taxonomy/taxonomy_summary.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("evaluations/accv2026/coverage_stride_sweep")
OUT  = Path("evaluations/accv2026/e5_taxonomy")
OUT.mkdir(parents=True, exist_ok=True)

MODELS   = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
            "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens", "finegym"]

summary_rows = []

for dataset in DATASETS:
    all_class_rows = []

    for model in MODELS:
        sweep_dir = BASE / f"{model}_{dataset}"
        f1  = sweep_dir / "cov100_s1_samples.csv"
        f16 = sweep_dir / "cov100_s16_samples.csv"
        if not f1.exists() or not f16.exists():
            continue

        df1  = pd.read_csv(f1)
        df16 = pd.read_csv(f16)

        # Filter valid samples
        def clean(df):
            return df[df["error"].isna() & ~df["skipped"].astype(bool)].copy()
        df1  = clean(df1)
        df16 = clean(df16)

        if df1.empty or df16.empty:
            continue

        # Per-class accuracy at stride=1 and stride=16
        acc1  = df1.groupby("label_id")["correct_top1"].mean().rename("acc_s1")
        acc16 = df16.groupby("label_id")["correct_top1"].mean().rename("acc_s16")
        merged = pd.concat([acc1, acc16], axis=1).dropna()
        merged["model"]   = model
        merged["dataset"] = dataset
        # Relative drop: how much of stride=1 accuracy was lost
        merged["abs_drop"] = merged["acc_s1"] - merged["acc_s16"]
        # Use abs_drop for taxonomy (rel_drop blows up when acc_s1 near 0)
        merged["rel_drop"] = merged["abs_drop"].clip(-1, 1)
        all_class_rows.append(merged.reset_index())

    if not all_class_rows:
        print(f"  {dataset}: no data")
        continue

    all_classes = pd.concat(all_class_rows, ignore_index=True)

    # Average sensitivity across all models for each class
    class_agg = all_classes.groupby("label_id").agg(
        n_models    = ("model", "nunique"),
        mean_acc_s1 = ("acc_s1",  "mean"),
        mean_acc_s16= ("acc_s16", "mean"),
        mean_rel_drop=("rel_drop","mean"),
        mean_abs_drop=("abs_drop","mean"),
        std_rel_drop= ("rel_drop","std"),
    ).reset_index()

    # Only include classes with data from ≥3 models
    class_agg = class_agg[class_agg["n_models"] >= 3].copy()

    # Taxonomy thresholds (tertile split of rel_drop)
    q33 = class_agg["mean_rel_drop"].quantile(0.33)
    q67 = class_agg["mean_rel_drop"].quantile(0.67)

    def classify(v):
        if v >= q67:   return "High"
        elif v >= q33: return "Moderate"
        else:          return "Low"

    class_agg["sensitivity"] = class_agg["mean_rel_drop"].apply(classify)
    class_agg["dataset"] = dataset

    class_agg.to_csv(OUT / f"{dataset}_class_taxonomy.csv", index=False)

    # Summary stats per tier
    for tier in ["High", "Moderate", "Low"]:
        sub = class_agg[class_agg["sensitivity"] == tier]
        summary_rows.append({
            "dataset": dataset,
            "tier": tier,
            "n_classes": len(sub),
            "mean_acc_s1":    sub["mean_acc_s1"].mean(),
            "mean_acc_s16":   sub["mean_acc_s16"].mean(),
            "mean_rel_drop":  sub["mean_rel_drop"].mean(),
            "mean_abs_drop_pp": sub["mean_abs_drop"].mean() * 100,
            "threshold_low":  q33,
            "threshold_high": q67,
        })

    print(f"  {dataset}: {len(class_agg)} classes → "
          f"High={len(class_agg[class_agg['sensitivity']=='High'])} "
          f"Mod={len(class_agg[class_agg['sensitivity']=='Moderate'])} "
          f"Low={len(class_agg[class_agg['sensitivity']=='Low'])}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "taxonomy_summary.csv", index=False)

print("\n" + "="*75)
print("E5 TAXONOMY — Action sensitivity tiers per dataset")
print("="*75)
print(f"\n{'Dataset':<15} {'Tier':<10} {'n_cls':>6} {'acc@s1':>8} {'acc@s16':>8} {'drop':>8}")
print("-"*60)
for _, r in summary_df.sort_values(["dataset","tier"]).iterrows():
    print(f"{r['dataset']:<15} {r['tier']:<10} {r['n_classes']:>6} "
          f"{r['mean_acc_s1']:>8.1%} {r['mean_acc_s16']:>8.1%} "
          f"{r['mean_abs_drop_pp']:>7.1f}pp")
    if r['tier'] == 'Low':
        print()
