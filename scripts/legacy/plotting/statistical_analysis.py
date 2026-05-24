#!/usr/bin/env python3
"""
Statistical Analysis of UCF-101 Temporal Sampling Results

Performs comprehensive statistical hypothesis testing and effect size computation
to validate key findings regarding temporal aliasing in action recognition.
"""


import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import argparse

# ============================================================
# ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description="Statistical analysis for temporal sampling results (supports all models)")
parser.add_argument('--csv', type=str, required=True, help='Path to main results CSV (temporal_sampling)')
parser.add_argument('--per-class-csv', type=str, required=True, help='Path to per-class results CSV')
parser.add_argument('--out-dir', type=str, default=None, help='Directory to save outputs (default: CSV parent)')
args = parser.parse_args()

main_csv = Path(args.csv)
per_class_csv = Path(args.per_class_csv)
out_dir = Path(args.out_dir) if args.out_dir else main_csv.parent
out_dir.mkdir(parents=True, exist_ok=True)

df_agg = pd.read_csv(main_csv)
df_per_class = pd.read_csv(per_class_csv)
# Deduplicate any repeated per-class entries (keep first occurrence).
# Some evaluation outputs contained duplicate rows for class×coverage×stride which
# inflate sample sizes and distort ANOVA/Levene statistics. Removing duplicates
# ensures correct degrees of freedom and unbiased effect-size estimates.
df_per_class = df_per_class.drop_duplicates(['class', 'coverage', 'stride'])

print("="*70)
print("STATISTICAL ANALYSIS: TEMPORAL SAMPLING EFFECTS ON ACTION RECOGNITION")
print("="*70)

# ============================================================
# 1. HYPOTHESIS TEST: Coverage Effect (ANOVA)
# ============================================================
print("\n[1] HYPOTHESIS TEST: Coverage Effect on Accuracy (One-way ANOVA)")
print("-" * 70)

groups_by_coverage = {}
for cov in sorted(df_per_class["coverage"].unique()):
    groups_by_coverage[cov] = df_per_class[
        (df_per_class["coverage"] == cov) & (df_per_class["stride"] == 8)
    ]["accuracy"].values

f_stat, p_value = stats.f_oneway(*groups_by_coverage.values())

print(f"Null Hypothesis: Accuracy is independent of frame coverage")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.2e}")
print(f"Result: {'REJECT H0 (coverage has significant effect)' if p_value < 0.05 else 'FAIL TO REJECT H0'}")

# Effect size (eta-squared)
all_accuracies = np.concatenate(list(groups_by_coverage.values()))
grand_mean = all_accuracies.mean()
ss_between = sum(
    len(group) * (group.mean() - grand_mean)**2 
    for group in groups_by_coverage.values()
)
ss_total = sum((acc - grand_mean)**2 for acc in all_accuracies)
eta_squared = ss_between / ss_total

print(f"Effect Size (η²): {eta_squared:.4f}")
print(f"Interpretation: {['Small', 'Medium', 'Large'][min(2, int(eta_squared/0.06))]}")

# ============================================================
# 2. PAIRWISE COMPARISONS: Coverage levels (t-tests with Bonferroni correction)
# ============================================================
print("\n[2] PAIRWISE COMPARISONS: Coverage Levels (Welch's t-test, Bonferroni corrected)")
print("-" * 70)

coverages = sorted(df_per_class["coverage"].unique())
n_comparisons = len(coverages) * (len(coverages) - 1) // 2
alpha_bonferroni = 0.05 / n_comparisons

print(f"Number of pairwise comparisons: {n_comparisons}")
print(f"Bonferroni-corrected α: {alpha_bonferroni:.4f}\n")

pairwise_results = []
for i, cov1 in enumerate(coverages):
    for cov2 in coverages[i+1:]:
        acc1 = df_per_class[
            (df_per_class["coverage"] == cov1) & (df_per_class["stride"] == 8)
        ]["accuracy"].values
        acc2 = df_per_class[
            (df_per_class["coverage"] == cov2) & (df_per_class["stride"] == 8)
        ]["accuracy"].values
        
        t_stat, p_val = stats.ttest_ind(acc1, acc2, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(acc1)-1)*acc1.std()**2 + (len(acc2)-1)*acc2.std()**2) / (len(acc1) + len(acc2) - 2))
        cohens_d = (acc1.mean() - acc2.mean()) / pooled_std
        
        pairwise_results.append({
            "Comparison": f"{int(cov1)}% vs {int(cov2)}%",
            "Mean Diff": acc1.mean() - acc2.mean(),
            "t-statistic": t_stat,
            "p-value": p_val,
            "Bonferroni Sig": "***" if p_val < alpha_bonferroni else "ns",
            "Cohen's d": cohens_d
        })

df_pairwise = pd.DataFrame(pairwise_results)
print(df_pairwise.to_string(index=False))

# ============================================================
# 3. STRIDE EFFECT (ANOVA at full coverage)
# ============================================================
print("\n[3] HYPOTHESIS TEST: Stride Effect at Full Coverage (One-way ANOVA)")
print("-" * 70)

full_cov_data = df_per_class[df_per_class["coverage"] == 100]
groups_by_stride = {}
for stride in sorted(full_cov_data["stride"].unique()):
    groups_by_stride[stride] = full_cov_data[
        full_cov_data["stride"] == stride
    ]["accuracy"].values

f_stat_stride, p_value_stride = stats.f_oneway(*groups_by_stride.values())

print(f"Null Hypothesis: Accuracy is independent of stride at full coverage")
print(f"F-statistic: {f_stat_stride:.4f}")
print(f"p-value: {p_value_stride:.2e}")
print(f"Result: {'REJECT H0 (stride has significant effect)' if p_value_stride < 0.05 else 'FAIL TO REJECT H0'}")

all_stride_accs = np.concatenate(list(groups_by_stride.values()))
grand_mean_stride = all_stride_accs.mean()
ss_between_stride = sum(
    len(group) * (group.mean() - grand_mean_stride)**2 
    for group in groups_by_stride.values()
)
ss_total_stride = sum((acc - grand_mean_stride)**2 for acc in all_stride_accs)
eta_squared_stride = ss_between_stride / ss_total_stride

print(f"Effect Size (η²): {eta_squared_stride:.4f}")

# ============================================================
# 4. ALIASING SENSITIVITY: Within-class correlation structure
# ============================================================
print("\n[4] ALIASING SENSITIVITY: Correlation Analysis")
print("-" * 70)

# Compute aliasing drop per class
aliasing_summary = (
    df_per_class[df_per_class["coverage"].isin([25, 100])]
    .pivot_table(index="class", columns="coverage", values="accuracy")
    .reset_index()
)
aliasing_summary["aliasing_drop"] = aliasing_summary[100] - aliasing_summary[25]

mean_drop = aliasing_summary["aliasing_drop"].mean()
std_drop = aliasing_summary["aliasing_drop"].std()
max_drop = aliasing_summary["aliasing_drop"].max()
min_drop = aliasing_summary["aliasing_drop"].min()

print(f"Mean aliasing drop (100% → 25% coverage): {mean_drop:.4f} ({mean_drop*100:.2f}%)")
print(f"Std dev: {std_drop:.4f}")
print(f"Range: [{min_drop:.4f}, {max_drop:.4f}]")
print(f"Coefficient of Variation: {std_drop/mean_drop:.4f}")

# Heterogeneity in aliasing sensitivity
print(f"\nVariance in aliasing drop across classes: {aliasing_summary['aliasing_drop'].var():.6f}")
print(f"→ Indicates HIGH heterogeneity in class-specific aliasing sensitivity")

# ============================================================
# 5. VARIANCE ANALYSIS: Homogeneity of variance test
# ============================================================
print("\n[5] HOMOGENEITY OF VARIANCE: Levene's Test")
print("-" * 70)

groups_for_levene = [
    df_per_class[
        (df_per_class["coverage"] == cov) & (df_per_class["stride"] == 8)
    ]["accuracy"].values
    for cov in sorted(df_per_class["coverage"].unique())
]

stat_levene, p_levene = stats.levene(*groups_for_levene)

print(f"Null Hypothesis: Variance is equal across coverage levels")
print(f"Levene's statistic: {stat_levene:.4f}")
print(f"p-value: {p_levene:.2e}")
print(f"Result: {'REJECT H0 (variances differ significantly)' if p_levene < 0.05 else 'FAIL TO REJECT H0'}")

# Variance by coverage
print("\nVariance by Coverage Level (Stride=8):")
for cov in sorted(df_per_class["coverage"].unique()):
    var_cov = df_per_class[
        (df_per_class["coverage"] == cov) & (df_per_class["stride"] == 8)
    ]["accuracy"].var()
    print(f"  {int(cov):3d}%: {var_cov:.6f}")

# ============================================================
# 6. EFFECT SIZE ANALYSIS: Aliasing vs Stride
# ============================================================
print("\n[6] EFFECT SIZE COMPARISON: Aliasing vs Stride Impact")
print("-" * 70)

# Aliasing impact: 100% vs 10%
acc_100_s8 = df_per_class[(df_per_class["coverage"] == 100) & (df_per_class["stride"] == 8)]["accuracy"]
acc_10_s8 = df_per_class[(df_per_class["coverage"] == 10) & (df_per_class["stride"] == 8)]["accuracy"]
aliasing_cohens_d = (acc_100_s8.mean() - acc_10_s8.mean()) / np.sqrt((acc_100_s8.var() + acc_10_s8.var()) / 2)

# Stride impact: stride-1 vs stride-16 at full coverage
acc_s1_100 = df_per_class[(df_per_class["coverage"] == 100) & (df_per_class["stride"] == 1)]["accuracy"]
acc_s16_100 = df_per_class[(df_per_class["coverage"] == 100) & (df_per_class["stride"] == 16)]["accuracy"]
stride_cohens_d = (acc_s1_100.mean() - acc_s16_100.mean()) / np.sqrt((acc_s1_100.var() + acc_s16_100.var()) / 2)

print(f"Aliasing Effect (100% vs 10% coverage, stride=8):")
print(f"  Mean difference: {(acc_100_s8.mean() - acc_10_s8.mean())*100:.2f}%")
print(f"  Cohen's d: {aliasing_cohens_d:.4f} ({'Large' if abs(aliasing_cohens_d) > 0.8 else 'Medium' if abs(aliasing_cohens_d) > 0.5 else 'Small'})")

print(f"\nStride Effect (stride-1 vs stride-16, coverage=100%):")
print(f"  Mean difference: {(acc_s1_100.mean() - acc_s16_100.mean())*100:.2f}%")
print(f"  Cohen's d: {stride_cohens_d:.4f} ({'Large' if abs(stride_cohens_d) > 0.8 else 'Medium' if abs(stride_cohens_d) > 0.5 else 'Small'})")

# ============================================================
# 7. SUMMARY TABLE FOR PAPER
# ============================================================
print("\n[7] SUMMARY STATISTICS TABLE")
print("-" * 70)

summary_stats = []
for cov in sorted(df_per_class["coverage"].unique()):
    cov_data = df_per_class[
        (df_per_class["coverage"] == cov) & (df_per_class["stride"] == 8)
    ]["accuracy"]
    
    summary_stats.append({
        "Coverage": f"{int(cov)}%",
        "N": len(cov_data),
        "Mean": cov_data.mean(),
        "Std": cov_data.std(),
        "Min": cov_data.min(),
        "Max": cov_data.max(),
        "95% CI": f"[{cov_data.mean() - 1.96*cov_data.sem():.4f}, {cov_data.mean() + 1.96*cov_data.sem():.4f}]"
    })

df_summary = pd.DataFrame(summary_stats)
print(df_summary.to_string(index=False))

# ============================================================
# 8. EXPORT RESULTS
# ============================================================
print("\n" + "="*70)
print("SAVING STATISTICAL RESULTS")
print("="*70)

# Save comprehensive results
stats_output = {
    "anova_coverage": {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared)
    },
    "anova_stride": {
        "f_statistic": float(f_stat_stride),
        "p_value": float(p_value_stride),
        "eta_squared": float(eta_squared_stride)
    },
    "aliasing_sensitivity": {
        "mean_drop": float(mean_drop),
        "std_drop": float(std_drop),
        "max_drop": float(max_drop),
        "min_drop": float(min_drop)
    },
    "variance_homogeneity": {
        "levene_statistic": float(stat_levene),
        "p_value": float(p_levene)
    },
    "effect_sizes": {
        "aliasing_cohens_d": float(aliasing_cohens_d),
        "stride_cohens_d": float(stride_cohens_d)
    }
}

with open(out_dir / "statistical_results.json", "w") as f:
    json.dump(stats_output, f, indent=2)


# Save pairwise comparison results
df_pairwise.to_csv(out_dir / "pairwise_coverage_comparisons.csv", index=False)

# Save summary statistics
df_summary.to_csv(out_dir / "summary_statistics_by_coverage.csv", index=False)

print("\n✅ Saved:")
print(f"  • {out_dir}/statistical_results.json")
print(f"  • {out_dir}/pairwise_coverage_comparisons.csv")
print(f"  • {out_dir}/summary_statistics_by_coverage.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
