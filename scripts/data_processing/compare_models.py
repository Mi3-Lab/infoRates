#!/usr/bin/env python3
"""
Multi-Model Comparison Analysis

Compares temporal aliasing effects across different models.
Generates comparative visualizations and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

def load_model_results(results_dir, model_name):
    """Load results for a specific model."""
    results_file = Path(results_dir) / f"results_{model_name}.csv"
    if not results_file.exists():
        print(f"⚠️ {results_file} not found")
        return None
    return pd.read_csv(results_file)

def compare_models_anova(timesformer_df, videomae_df, vivit_df):
    """
    Perform ANOVA to test if model differences are significant.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS TEST: Model Effect on Accuracy")
    print("="*70)
    
    # Compare best accuracy by model
    ts_best = timesformer_df["accuracy"].max()
    vm_best = videomae_df["accuracy"].max()
    vt_best = vivit_df["accuracy"].max()
    
    print(f"\nBest Accuracy by Model:")
    print(f"  TimeSformer: {ts_best:.4f}")
    print(f"  VideoMAE:    {vm_best:.4f}")
    print(f"  ViViT:       {vt_best:.4f}")
    
    # ANOVA on full accuracies
    f_stat, p_val = stats.f_oneway(
        timesformer_df["accuracy"].values,
        videomae_df["accuracy"].values,
        vivit_df["accuracy"].values,
    )
    
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_val:.4e}")
    print(f"  Result: {'SIGNIFICANT model effect' if p_val < 0.05 else 'No significant model effect'}")
    
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "timesformer_best": float(ts_best),
        "videomae_best": float(vm_best),
        "vivit_best": float(vt_best),
    }

def analyze_aliasing_robustness(timesformer_df, videomae_df, vivit_df):
    """
    Compare aliasing sensitivity across models.
    Analyze how much each model drops from 100% → 25% coverage.
    """
    print("\n" + "="*70)
    print("ALIASING ROBUSTNESS COMPARISON")
    print("="*70)
    
    results = {}
    
    for name, df in [("TimeSformer", timesformer_df), ("VideoMAE", videomae_df), ("ViViT", vivit_df)]:
        # At 100% coverage, stride=8
        acc_100 = df[(df["coverage"] == 100) & (df["stride"] == 8)]["accuracy"].values
        # At 25% coverage, stride=8
        acc_25 = df[(df["coverage"] == 25) & (df["stride"] == 8)]["accuracy"].values
        
        if len(acc_100) > 0 and len(acc_25) > 0:
            mean_100 = acc_100.mean()
            mean_25 = acc_25.mean()
            drop = mean_100 - mean_25
            drop_pct = (drop / mean_100) * 100 if mean_100 > 0 else 0
            
            results[name] = {
                "accuracy_100pct": float(mean_100),
                "accuracy_25pct": float(mean_25),
                "drop_absolute": float(drop),
                "drop_percentage": float(drop_pct),
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy @ 100% coverage: {mean_100:.4f}")
            print(f"  Accuracy @  25% coverage: {mean_25:.4f}")
            print(f"  Drop: {drop:.4f} ({drop_pct:.2f}%)")
            
            # Robustness interpretation
            if drop_pct < 5:
                robustness = "VERY HIGH (< 5%)"
            elif drop_pct < 10:
                robustness = "HIGH (5-10%)"
            elif drop_pct < 15:
                robustness = "MODERATE (10-15%)"
            elif drop_pct < 25:
                robustness = "LOW (15-25%)"
            else:
                robustness = "VERY LOW (> 25%)"
            
            print(f"  Robustness: {robustness}")
            results[name]["robustness_level"] = robustness
    
    return results

def plot_model_comparison(timesformer_df, videomae_df, vivit_df, output_dir):
    """Generate comparison visualizations."""
    
    # Combine all results
    combined = pd.concat([
        timesformer_df.assign(model="TimeSformer"),
        videomae_df.assign(model="VideoMAE"),
        vivit_df.assign(model="ViViT"),
    ], ignore_index=True)
    
    # 1. Accuracy vs Coverage by Model (stride=8)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in ["TimeSformer", "VideoMAE", "ViViT"]:
        data = combined[(combined["model"] == model) & (combined["stride"] == 8)].sort_values("coverage")
        ax.plot(data["coverage"], data["accuracy"], marker="o", linewidth=2.5, label=model)
    
    ax.set_xlabel("Frame Coverage (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Multi-Model Comparison: Aliasing Sensitivity (Stride=8)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.05)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison_accuracy_vs_coverage.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: comparison_accuracy_vs_coverage.png")
    
    # 2. Heatmap comparison (Coverage vs Stride) for each model
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for idx, model in enumerate(["TimeSformer", "VideoMAE", "ViViT"]):
        data = combined[combined["model"] == model]
        pivot = data.pivot(index="stride", columns="coverage", values="accuracy")
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=axes[idx],
            cbar_kws={"label": "Accuracy"},
            vmin=0.6,
            vmax=1.0,
        )
        axes[idx].set_title(f"{model}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Coverage (%)")
        axes[idx].set_ylabel("Stride")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: comparison_heatmaps.png")
    
    # 3. Best accuracy per model (bar chart)
    best_accs = []
    for model in ["TimeSformer", "VideoMAE", "ViViT"]:
        data = combined[combined["model"] == model]
        best = data.loc[data["accuracy"].idxmax()]
        best_accs.append({
            "Model": model,
            "Best Accuracy": best["accuracy"],
            "Coverage": int(best["coverage"]),
            "Stride": int(best["stride"]),
        })
    
    best_df = pd.DataFrame(best_accs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(best_df["Model"], best_df["Best Accuracy"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax.set_ylabel("Best Accuracy", fontsize=12)
    ax.set_title("Best Accuracy Comparison Across Models", fontsize=13, fontweight="bold")
    ax.set_ylim(0.95, 1.001)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison_best_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: comparison_best_accuracy.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare multi-model results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/UCF101_data/results",
        help="Directory containing results files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/UCF101_data/results",
        help="Directory for output plots and analysis",
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MULTI-MODEL COMPARISON ANALYSIS")
    print("="*70)
    
    # Load results for all models
    print("\nLoading results...")
    ts_df = load_model_results(results_dir, "timesformer")
    vm_df = load_model_results(results_dir, "videomae")
    vt_df = load_model_results(results_dir, "vivit")
    
    if ts_df is None or vm_df is None or vt_df is None:
        print("✗ Missing results for one or more models")
        return
    
    print(f"✓ TimeSformer: {len(ts_df)} configurations")
    print(f"✓ VideoMAE: {len(vm_df)} configurations")
    print(f"✓ ViViT: {len(vt_df)} configurations")
    
    # Statistical comparison
    anova_results = compare_models_anova(ts_df, vm_df, vt_df)
    
    # Aliasing robustness analysis
    aliasing_results = analyze_aliasing_robustness(ts_df, vm_df, vt_df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_model_comparison(ts_df, vm_df, vt_df, output_dir)
    
    # Save analysis summary
    analysis_summary = {
        "statistical_comparison": anova_results,
        "aliasing_robustness": aliasing_results,
    }
    
    with open(output_dir / "multimodel_analysis.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\n✓ Analysis saved: multimodel_analysis.json")
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
