#!/usr/bin/env python3
"""
Generate comparison plot: Baseline vs TRA Robustness
Plots heatmaps and degradation curves similar to other analysis plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_baseline_tra_comparison_plot(output_dir="docs/images/comparative"):
    """
    Create degradation curves plot comparing Baseline vs TRA robustness.
    Includes both absolute degradation and percentage degradation labels.
    """
    
    # Data from your comparison results
    data = {
        'coverage': [25, 25, 25, 25, 25, 50, 50, 50, 50, 50, 75, 75, 75, 75, 75, 100, 100, 100, 100, 100],
        'stride': [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16],
        'baseline': [0.8406, 0.8222, 0.8143, 0.7160, 0.7160, 0.8608, 0.8610, 0.8432, 0.8296, 0.7160, 
                     0.8687, 0.8682, 0.8529, 0.8537, 0.8377, 0.8718, 0.8693, 0.8694, 0.8537, 0.8377],
        'tra': [0.8515, 0.8449, 0.8426, 0.8238, 0.8238, 0.8583, 0.8570, 0.8523, 0.8459, 0.8238,
                0.8628, 0.8619, 0.8547, 0.8536, 0.8489, 0.8660, 0.8624, 0.8639, 0.8536, 0.8489]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate metrics
    df['absolute_improvement'] = df['tra'] - df['baseline']
    df['relative_improvement'] = (df['absolute_improvement'] / df['baseline']) * 100
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Temporal Robustness: Baseline vs TRA Degradation (% Loss)', 
                 fontsize=17, fontweight='bold', y=0.98)
    
    # Get full-data reference points
    baseline_ref = df[(df['coverage'] == 100) & (df['stride'] == 1)]['baseline'].values[0]
    tra_ref = df[(df['coverage'] == 100) & (df['stride'] == 1)]['tra'].values[0]
    
    colors_baseline = ['#C62828', '#E64A19', '#FF6F00']  # Reds for baseline
    colors_tra = ['#1976D2', '#00ACC1', '#00897B']       # Blues for TRA
    
    stride_values = sorted(df['stride'].unique())
    
    # Color palette for all strides
    colors_base = ['#C62828', '#E64A19', '#FF6F00', '#F57C00', '#FB8C00']  # Reds for baseline
    colors_tra_palette = ['#1976D2', '#00ACC1', '#00897B', '#0097A7', '#006064']  # Blues for TRA
    
    # Track line objects for later annotation
    lines_data = []
    
    for idx, stride in enumerate(stride_values):  # Plot all strides
        subset = df[df['stride'] == stride].sort_values('coverage')
        
        # Baseline degradation in percentage
        baseline_deg = baseline_ref - subset['baseline'].values
        baseline_deg_pct = (baseline_deg / baseline_ref) * 100
        
        ax.plot(
            subset['coverage'].values,
            baseline_deg_pct,
            marker='o',
            linestyle='--',
            linewidth=2.8,
            markersize=9,
            label=f'Baseline (stride={stride})',
            color=colors_base[idx],
            alpha=0.75,
        )
        
        # TRA degradation in percentage
        tra_deg = tra_ref - subset['tra'].values
        tra_deg_pct = (tra_deg / tra_ref) * 100
        
        ax.plot(
            subset['coverage'].values,
            tra_deg_pct,
            marker='s',
            linestyle='-',
            linewidth=3.0,
            markersize=9,
            label=f'TRA (stride={stride})',
            color=colors_tra_palette[idx],
            alpha=0.9,
        )
        
        # Store data for annotations
        lines_data.append({
            'subset': subset,
            'baseline_deg_pct': baseline_deg_pct,
            'tra_deg_pct': tra_deg_pct,
            'color_baseline': colors_base[idx],
            'color_tra': colors_tra_palette[idx],
        })
    
    # Add percentage annotations
    for data_item in lines_data:
        subset = data_item['subset']
        
        # Annotate baseline with percentages
        for i, (cov, baseline_pct) in enumerate(zip(subset['coverage'].values, data_item['baseline_deg_pct'])):
            ax.annotate(
                f'{baseline_pct:.1f}%',
                xy=(cov, baseline_pct),
                xytext=(-14, -12),
                textcoords='offset points',
                fontsize=7.5,
                color=data_item['color_baseline'],
                fontweight='bold',
                alpha=0.75,
            )
        
        # Annotate TRA with percentages
        for i, (cov, tra_pct) in enumerate(zip(subset['coverage'].values, data_item['tra_deg_pct'])):
            ax.annotate(
                f'{tra_pct:.1f}%',
                xy=(cov, tra_pct),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=7.5,
                color=data_item['color_tra'],
                fontweight='bold',
                alpha=0.85,
            )
    
    ax.set_xlabel('Frame Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy Degradation (%)', fontsize=14, fontweight='bold')
    
    # Get legend handles and labels, then reorder them to match the line drawing order
    handles, labels = ax.get_legend_handles_labels()
    # Sort by label to group Baselines and TRAs together (stride order)
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: (x[0].split('=')[1].rstrip(')'), 'Baseline' in x[0]))
    sorted_labels, sorted_handles = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    ax.legend(sorted_handles, sorted_labels, fontsize=9.5, loc='upper right', framealpha=0.96, edgecolor='#333333', ncol=2, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
    ax.axhline(0, color='#333333', linestyle='-', linewidth=1.2, alpha=0.6)
    ax.set_ylim([-1, 20])
    ax.set_xlim([18, 105])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_file = output_path / "baseline_tra_degradation_curves.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved degradation curves plot to {save_file}")
    
    plt.close()
    
    return df


def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"Mean Absolute Improvement: {df['absolute_improvement'].mean():+.4f}")
    print(f"Mean Relative Improvement: {df['relative_improvement'].mean():+.2f}%")
    print(f"Max Improvement: {df['absolute_improvement'].max():+.4f} "
          f"@ coverage={df.loc[df['absolute_improvement'].idxmax(), 'coverage']}, "
          f"stride={df.loc[df['absolute_improvement'].idxmax(), 'stride']}")
    
    low_cov = df[df['coverage'] <= 50]
    if len(low_cov) > 0:
        print(f"\nLow Coverage (≤50%) Mean Improvement: {low_cov['absolute_improvement'].mean():+.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    df = create_baseline_tra_comparison_plot()
    print_summary_stats(df)
