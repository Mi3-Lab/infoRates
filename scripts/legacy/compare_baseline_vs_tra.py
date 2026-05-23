"""
Compare baseline vs TRA training results.

Analyzes robustness improvements from Temporal Robustness Augmentation (TRA)
by comparing accuracy degradation across coverage×stride grid.

Usage:
    python scripts/compare_baseline_vs_tra.py --model timesformer
    
    # Generate comparison plots
    python scripts/compare_baseline_vs_tra.py --model timesformer --plot
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def load_robustness_results(model: str, mode: str, base_dir: str = "fine_tuned_models/tra_experiments") -> Dict[str, float]:
    """Load robustness results JSON."""
    results_path = Path(base_dir) / mode / f"robustness_{model}.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def parse_results_to_dataframe(results: Dict[str, float], mode: str) -> pd.DataFrame:
    """
    Convert robustness results to DataFrame.
    
    Input format: {"cov25_stride1": 0.85, "cov25_stride2": 0.78, ...}
    Output: DataFrame with columns [coverage, stride, accuracy, mode]
    """
    rows = []
    for key, acc in results.items():
        # Parse key like "cov25_stride1"
        parts = key.split('_')
        coverage = int(parts[0].replace('cov', ''))
        stride = int(parts[1].replace('stride', ''))
        
        rows.append({
            'coverage': coverage,
            'stride': stride,
            'accuracy': acc,
            'mode': mode,
        })
    
    return pd.DataFrame(rows)


def calculate_degradation_metrics(baseline_df: pd.DataFrame, tra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accuracy degradation metrics.
    
    Returns DataFrame with absolute degradation and relative improvement.
    """
    # Merge baseline and TRA results
    merged = baseline_df.merge(
        tra_df,
        on=['coverage', 'stride'],
        suffixes=('_baseline', '_tra')
    )
    
    # Get baseline accuracy at 100% coverage, stride 1
    baseline_100 = baseline_df[
        (baseline_df['coverage'] == 100) & (baseline_df['stride'] == 1)
    ]['accuracy'].values[0]
    
    tra_100 = tra_df[
        (tra_df['coverage'] == 100) & (tra_df['stride'] == 1)
    ]['accuracy'].values[0]
    
    # Calculate metrics
    merged['baseline_degradation'] = baseline_100 - merged['accuracy_baseline']
    merged['tra_degradation'] = tra_100 - merged['accuracy_tra']
    merged['absolute_improvement'] = merged['accuracy_tra'] - merged['accuracy_baseline']
    merged['relative_improvement'] = (
        (merged['accuracy_tra'] - merged['accuracy_baseline']) / merged['accuracy_baseline']
    ) * 100  # Percentage
    
    # Degradation reduction (negative means TRA has less degradation)
    merged['degradation_reduction'] = merged['tra_degradation'] - merged['baseline_degradation']
    
    return merged


def print_comparison_table(comparison_df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print("Baseline vs TRA Robustness Comparison")
    print("="*90)
    print(f"{'Coverage':<10} {'Stride':<8} {'Baseline':<10} {'TRA':<10} {'Abs Δ':<10} {'Rel Δ (%)':<12}")
    print("-"*90)
    
    for _, row in comparison_df.iterrows():
        print(
            f"{row['coverage']:<10} "
            f"{row['stride']:<8} "
            f"{row['accuracy_baseline']:.4f}    "
            f"{row['accuracy_tra']:.4f}    "
            f"{row['absolute_improvement']:+.4f}    "
            f"{row['relative_improvement']:+.2f}"
        )
    
    print("-"*90)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Mean Absolute Improvement: {comparison_df['absolute_improvement'].mean():+.4f}")
    print(f"  Mean Relative Improvement: {comparison_df['relative_improvement'].mean():+.2f}%")
    print(f"  Max Absolute Improvement: {comparison_df['absolute_improvement'].max():+.4f} "
          f"(coverage={comparison_df.loc[comparison_df['absolute_improvement'].idxmax(), 'coverage']}, "
          f"stride={comparison_df.loc[comparison_df['absolute_improvement'].idxmax(), 'stride']})")
    
    # Highlight low-coverage performance
    low_coverage = comparison_df[comparison_df['coverage'] <= 50]
    if len(low_coverage) > 0:
        print(f"\nLow Coverage (≤50%) Improvement: {low_coverage['absolute_improvement'].mean():+.4f}")
    
    print("="*90 + "\n")


def plot_comparison(comparison_df: pd.DataFrame, model: str, save_dir: str = "docs/figures"):
    """
    Generate comparison plots.
    
    Creates:
    1. Heatmap: Baseline accuracy
    2. Heatmap: TRA accuracy
    3. Heatmap: Absolute improvement
    4. Line plot: Degradation curves
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for heatmaps (pivot tables)
    baseline_pivot = comparison_df.pivot(
        index='stride', columns='coverage', values='accuracy_baseline'
    )
    tra_pivot = comparison_df.pivot(
        index='stride', columns='coverage', values='accuracy_tra'
    )
    improvement_pivot = comparison_df.pivot(
        index='stride', columns='coverage', values='absolute_improvement'
    )
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Baseline heatmap
    sns.heatmap(
        baseline_pivot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd_r',
        ax=axes[0, 0],
        cbar_kws={'label': 'Accuracy'},
        vmin=0, vmax=1,
    )
    axes[0, 0].set_title(f'Baseline Accuracy ({model.upper()})', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Coverage (%)')
    axes[0, 0].set_ylabel('Stride')
    
    # 2. TRA heatmap
    sns.heatmap(
        tra_pivot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd_r',
        ax=axes[0, 1],
        cbar_kws={'label': 'Accuracy'},
        vmin=0, vmax=1,
    )
    axes[0, 1].set_title(f'TRA Accuracy ({model.upper()})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Coverage (%)')
    axes[0, 1].set_ylabel('Stride')
    
    # 3. Improvement heatmap
    sns.heatmap(
        improvement_pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        ax=axes[1, 0],
        cbar_kws={'label': 'Absolute Improvement'},
        center=0,
    )
    axes[1, 0].set_title('Absolute Improvement (TRA - Baseline)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Coverage (%)')
    axes[1, 0].set_ylabel('Stride')
    
    # 4. Degradation curves
    baseline_100 = comparison_df[
        (comparison_df['coverage'] == 100) & (comparison_df['stride'] == 1)
    ]['accuracy_baseline'].values[0]
    tra_100 = comparison_df[
        (comparison_df['coverage'] == 100) & (comparison_df['stride'] == 1)
    ]['accuracy_tra'].values[0]
    
    for stride in sorted(comparison_df['stride'].unique()):
        subset = comparison_df[comparison_df['stride'] == stride].sort_values('coverage')
        
        # Baseline degradation
        axes[1, 1].plot(
            subset['coverage'],
            baseline_100 - subset['accuracy_baseline'],
            marker='o',
            linestyle='--',
            label=f'Baseline (stride={stride})',
            alpha=0.7,
        )
        
        # TRA degradation
        axes[1, 1].plot(
            subset['coverage'],
            tra_100 - subset['accuracy_tra'],
            marker='s',
            linestyle='-',
            label=f'TRA (stride={stride})',
        )
    
    axes[1, 1].set_xlabel('Coverage (%)', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy Degradation', fontsize=12)
    axes[1, 1].set_title('Degradation Curves (Baseline vs TRA)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / f"tra_comparison_{model}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comparison plot to {save_path}")
    
    plt.close()


def generate_latex_table(comparison_df: pd.DataFrame, model: str) -> str:
    """
    Generate LaTeX table for paper.
    
    Returns formatted LaTeX table string.
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Robustness Comparison: Baseline vs TRA (" + model.upper() + ")}")
    latex.append("\\label{tab:tra_comparison_" + model + "}")
    latex.append("\\begin{tabular}{cccccc}")
    latex.append("\\toprule")
    latex.append("Coverage & Stride & Baseline & TRA & $\\Delta$ Abs & $\\Delta$ Rel (\\%) \\\\")
    latex.append("\\midrule")
    
    for _, row in comparison_df.iterrows():
        latex.append(
            f"{row['coverage']:.0f}\\% & "
            f"{row['stride']:.0f} & "
            f"{row['accuracy_baseline']:.3f} & "
            f"{row['accuracy_tra']:.3f} & "
            f"{row['absolute_improvement']:+.3f} & "
            f"{row['relative_improvement']:+.1f} \\\\"
        )
    
    latex.append("\\midrule")
    latex.append(
        f"\\multicolumn{{4}}{{l}}{{Mean Improvement:}} & "
        f"{comparison_df['absolute_improvement'].mean():+.3f} & "
        f"{comparison_df['relative_improvement'].mean():+.1f} \\\\"
    )
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs TRA")
    parser.add_argument("--model", type=str, required=True, choices=["timesformer", "videomae", "vivit"])
    parser.add_argument("--base-dir", type=str, default="fine_tuned_models/tra_experiments")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--save-latex", action="store_true", help="Save LaTeX table")
    parser.add_argument("--output-dir", type=str, default="docs/tables")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Comparing Baseline vs TRA for {args.model.upper()}")
    print(f"{'='*70}\n")
    
    # Load results
    print("Loading results...")
    baseline_results = load_robustness_results(args.model, mode="baseline", base_dir=args.base_dir)
    tra_results = load_robustness_results(args.model, mode="tra", base_dir=args.base_dir)
    
    # Convert to DataFrames
    baseline_df = parse_results_to_dataframe(baseline_results, mode="baseline")
    tra_df = parse_results_to_dataframe(tra_results, mode="tra")
    
    # Calculate comparison metrics
    comparison_df = calculate_degradation_metrics(baseline_df, tra_df)
    
    # Print table
    print_comparison_table(comparison_df)
    
    # Generate plots
    if args.plot:
        print("Generating comparison plots...")
        plot_comparison(comparison_df, args.model)
    
    # Save LaTeX table
    if args.save_latex:
        latex_table = generate_latex_table(comparison_df, args.model)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"tra_comparison_{args.model}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        print(f"💾 Saved LaTeX table to {output_path}")
    
    # Summary insights
    print("\n" + "="*70)
    print("Key Insights:")
    print("="*70)
    
    # Check if TRA helps more at low coverage
    low_cov = comparison_df[comparison_df['coverage'] <= 50]
    high_cov = comparison_df[comparison_df['coverage'] > 50]
    
    if len(low_cov) > 0 and len(high_cov) > 0:
        low_improvement = low_cov['absolute_improvement'].mean()
        high_improvement = high_cov['absolute_improvement'].mean()
        
        print(f"✅ Low Coverage (≤50%) Improvement: {low_improvement:+.4f}")
        print(f"✅ High Coverage (>50%) Improvement: {high_improvement:+.4f}")
        
        if low_improvement > high_improvement:
            print("   → TRA provides greater benefit when temporal sampling is aggressive")
    
    # Check stride impact
    high_stride = comparison_df[comparison_df['stride'] >= 2]
    if len(high_stride) > 0:
        high_stride_improvement = high_stride['absolute_improvement'].mean()
        print(f"✅ High Stride (≥2) Improvement: {high_stride_improvement:+.4f}")
        print("   → TRA reduces vulnerability to sparse temporal sampling")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
