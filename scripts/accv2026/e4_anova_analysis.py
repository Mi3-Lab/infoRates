"""E4 — Two-way ANOVA: Coverage × Stride effects on accuracy.

For each (model, dataset), tests:
  - Main effect of Coverage (η²_coverage)
  - Main effect of Stride    (η²_stride)
  - Interaction Coverage × Stride
  - Post-hoc pairwise comparisons (Tukey HSD) between stride levels

Outputs:
  evaluations/accv2026/e4_anova/anova_results.csv      — per model/dataset
  evaluations/accv2026/e4_anova/anova_summary.csv       — aggregate effect sizes
  evaluations/accv2026/e4_anova/posthoc_stride.csv      — pairwise stride comparisons
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

BASE = Path("evaluations/accv2026/coverage_stride_sweep")
OUT  = Path("evaluations/accv2026/e4_anova")
OUT.mkdir(parents=True, exist_ok=True)

MODELS   = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
            "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens", "finegym"]

def eta_squared(ss_effect, ss_total):
    return ss_effect / ss_total if ss_total > 0 else 0

def two_way_anova(df):
    """Simple two-way ANOVA using scipy F-tests on the sweep_summary data."""
    # Use per-config top1 accuracy (one observation per (cov, stride) cell)
    grand_mean = df["top1"].mean()
    ss_total = ((df["top1"] - grand_mean) ** 2).sum()

    # Main effect: Coverage
    cov_means = df.groupby("coverage")["top1"].mean()
    cov_counts = df.groupby("coverage")["top1"].count()
    ss_cov = ((cov_means - grand_mean) ** 2 * cov_counts).sum()

    # Main effect: Stride
    str_means = df.groupby("stride")["top1"].mean()
    str_counts = df.groupby("stride")["top1"].count()
    ss_str = ((str_means - grand_mean) ** 2 * str_counts).sum()

    # Residual
    ss_res = ss_total - ss_cov - ss_str

    # Degrees of freedom
    k_cov = df["coverage"].nunique() - 1
    k_str = df["stride"].nunique() - 1
    df_res = len(df) - df["coverage"].nunique() - df["stride"].nunique() + 1

    ms_cov = ss_cov / k_cov if k_cov > 0 else 0
    ms_str = ss_str / k_str if k_str > 0 else 0
    ms_res = ss_res / df_res if df_res > 0 else 1e-9

    f_cov = ms_cov / ms_res
    f_str = ms_str / ms_res

    p_cov = 1 - stats.f.cdf(f_cov, k_cov, df_res) if ms_res > 0 else 1.0
    p_str = 1 - stats.f.cdf(f_str, k_str, df_res) if ms_res > 0 else 1.0

    return {
        "ss_coverage": ss_cov, "ss_stride": ss_str, "ss_residual": ss_res, "ss_total": ss_total,
        "eta2_coverage": eta_squared(ss_cov, ss_total),
        "eta2_stride":   eta_squared(ss_str, ss_total),
        "f_coverage": f_cov, "p_coverage": p_cov,
        "f_stride":   f_str, "p_stride":   p_str,
        "n_configs":  len(df),
    }

anova_rows   = []
posthoc_rows = []

for model in MODELS:
    for dataset in DATASETS:
        sweep_csv = BASE / f"{model}_{dataset}" / "sweep_summary.csv"
        if not sweep_csv.exists():
            continue

        df = pd.read_csv(sweep_csv)
        if len(df) < 10:
            continue

        # Skip feature-collapse models (mean accuracy < 5%)
        if df["top1"].mean() < 0.05:
            print(f"  SKIP {model}/{dataset}: feature collapse (mean={df['top1'].mean():.3f})")
            continue

        print(f"  {model}/{dataset}: {len(df)} configs, mean acc={df['top1'].mean():.3f}")

        result = two_way_anova(df)
        result["model"]   = model
        result["dataset"] = dataset

        # Dominant effect classification
        if result["eta2_stride"] > result["eta2_coverage"]:
            result["dominant_effect"] = "STRIDE"
        else:
            result["dominant_effect"] = "COVERAGE"

        # Effect size interpretation (Cohen's conventions for η²)
        def interpret_eta2(e):
            if e < 0.01: return "negligible"
            if e < 0.06: return "small"
            if e < 0.14: return "medium"
            return "large"

        result["eta2_stride_interp"]   = interpret_eta2(result["eta2_stride"])
        result["eta2_coverage_interp"] = interpret_eta2(result["eta2_coverage"])
        anova_rows.append(result)

        # Post-hoc: pairwise stride comparisons at coverage=100%
        sub100 = df[df["coverage"] == 100].sort_values("stride")
        stride_accs = {int(s): float(a) for s, a in
                       zip(sub100["stride"], sub100["top1"])}

        for s1, s2 in combinations(sorted(stride_accs.keys()), 2):
            diff = (stride_accs[s1] - stride_accs[s2]) * 100  # pp
            posthoc_rows.append({
                "model": model, "dataset": dataset,
                "stride_a": s1, "stride_b": s2,
                "acc_a": stride_accs[s1] * 100,
                "acc_b": stride_accs[s2] * 100,
                "diff_pp": diff,
                "abs_diff_pp": abs(diff),
            })

# Save
anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUT / "anova_results.csv", index=False)

posthoc_df = pd.DataFrame(posthoc_rows)
posthoc_df.to_csv(OUT / "posthoc_stride.csv", index=False)

# Aggregate summary: avg η² per model across datasets
summary_rows = []
for model in MODELS:
    sub = anova_df[anova_df["model"] == model]
    if sub.empty: continue
    summary_rows.append({
        "model": model,
        "n_datasets": len(sub),
        "mean_eta2_stride":   sub["eta2_stride"].mean(),
        "mean_eta2_coverage": sub["eta2_coverage"].mean(),
        "pct_stride_dominant": (sub["dominant_effect"] == "STRIDE").mean() * 100,
        "mean_p_stride": sub["p_stride"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "anova_summary.csv", index=False)

# Print results
print("\n" + "="*80)
print("E4 ANOVA — Effect sizes (η²): Coverage vs Stride on accuracy")
print("η² interpretation: <0.01 negligible, <0.06 small, <0.14 medium, ≥0.14 large")
print("="*80)
print(f"\n{'Model':<16} {'Dataset':<14} | {'η²(stride)':>11} {'η²(cov)':>9} | {'Dominant':>9} {'p(stride)':>10}")
print("-"*75)
for _, row in anova_df.sort_values(["model","dataset"]).iterrows():
    interp = f"({row['eta2_stride_interp'][:3]})"
    print(f"{row['model']:<16} {row['dataset']:<14} | "
          f"{row['eta2_stride']:>9.3f}{interp} {row['eta2_coverage']:>9.3f} | "
          f"{row['dominant_effect']:>9} {row['p_stride']:>10.4f}")

print("\n" + "="*80)
print("AGGREGATE — Mean η² per model (across datasets)")
print("="*80)
print(f"\n{'Model':<16} | {'η²(stride)':>11} {'η²(cov)':>10} | {'% stride dominant':>18} | Interpretation")
print("-"*75)
for _, row in summary_df.sort_values("mean_eta2_stride", ascending=False).iterrows():
    interp = ("STRIDE dominates" if row["mean_eta2_stride"] > 0.14
              else "COVERAGE dominates" if row["mean_eta2_coverage"] > row["mean_eta2_stride"]
              else "mixed")
    print(f"{row['model']:<16} | {row['mean_eta2_stride']:>11.3f} {row['mean_eta2_coverage']:>10.3f} | "
          f"{row['pct_stride_dominant']:>17.0f}% | {interp}")
