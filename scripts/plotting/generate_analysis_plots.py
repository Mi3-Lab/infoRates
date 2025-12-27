#!/usr/bin/env python3
"""
Master Plotting Script for Temporal Aliasing Analysis

Generates all necessary plots and statistical analyses for a given model and dataset.
This consolidates multiple individual plotting scripts into one comprehensive analysis.

Usage:
    python scripts/plotting/plot_all.py --model videomae --dataset ucf101
    python scripts/plotting/plot_all.py --model all --dataset kinetics400
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def run_statistical_analysis(csv_path, per_class_csv_path, output_dir):
    """Run comprehensive statistical analysis."""
    print("🔬 Running statistical analysis...")

    import subprocess
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "statistical_analysis.py"),
            "--csv", str(csv_path),
            "--per-class-csv", str(per_class_csv_path),
            "--out-dir", str(output_dir)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Statistical analysis completed")
        else:
            print(f"⚠️  Statistical analysis failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Could not run statistical analysis: {e}")

def generate_distribution_plot(per_class_csv, output_dir, model_name, stride=8):
    """Generate per-class accuracy distribution plots (publication-quality).

    Styling decisions:
    - Use perceptually-uniform `viridis` palette across coverage levels
    - Display accuracies as percentages (1 decimal) on the y-axis
    - Remove per-point annotations; annotate only medians above each box/violin
    - Tight layout for clean integration into multi-panel composites
    """
    print("📊 Generating distribution plots (styled)")

    df = pd.read_csv(per_class_csv)

    # Filter by stride
    df = df[df['stride'] == stride]

    # Prepare data
    coverages = sorted(df['coverage'].unique())
    class_list = sorted(df['class'].unique())

    acc_matrix = np.zeros((len(class_list), len(coverages)))
    for i, cls in enumerate(class_list):
        for j, cov in enumerate(coverages):
            acc = df[(df['class'] == cls) & (df['coverage'] == cov)]['accuracy']
            acc_matrix[i, j] = (acc.values[0] * 100.0) if not acc.empty else np.nan

    # Convert to DataFrame with percent values
    acc_df = pd.DataFrame(acc_matrix, columns=coverages)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Color palette (viridis) with one color per coverage
    pal = sns.color_palette("viridis", n_colors=len(coverages))

    # Boxplot with cleaner style
    sns.boxplot(data=acc_df, ax=axes[0], palette=pal, fliersize=0, linewidth=1.2)
    axes[0].set_xlabel('Frame Coverage (%)', fontsize=14)
    axes[0].set_ylabel('Per-Class Accuracy (%)', fontsize=14)
    axes[0].set_title('Boxplot: Per-Class Accuracy by Coverage', fontsize=16)
    axes[0].set_xticklabels([str(int(c)) for c in coverages], fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.4)

    # Annotate medians above boxes
    medians = acc_df.median(axis=0)
    for i, m in enumerate(medians):
        axes[0].annotate(f"{m:.1f}%", xy=(i, m), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='semibold')

    # Violin plot with quartiles and same palette
    sns.violinplot(data=acc_df, ax=axes[1], inner='quartile', palette=pal, bw=0.2)
    axes[1].set_xlabel('Frame Coverage (%)', fontsize=14)
    axes[1].set_title('Violin: Per-Class Accuracy by Coverage', fontsize=16)
    axes[1].set_xticklabels([str(int(c)) for c in coverages], fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.4)

    # Annotate medians on violin plot
    for i, m in enumerate(medians):
        axes[1].annotate(f"{m:.1f}%", xy=(i, m), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='semibold')

    # Format y-axis as percent with one decimal
    import matplotlib.ticker as mtick
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    plt.suptitle(f'Distribution of Per-Class Accuracies at Stride-{stride} Across Coverage Levels ({model_name.capitalize()})', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save both per-model and a higher-resolution version for the composite
    output_path = output_dir / f"per_class_distribution_by_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Also save a slightly larger version for inclusion in composites
    output_path_hi = output_dir / f"per_class_distribution_by_coverage@2x.png"
    plt.savefig(output_path_hi, dpi=600, bbox_inches='tight')

    plt.close()
    print(f"✅ Saved: {output_path} and {output_path_hi}")

def compute_shared_classes(models_dirs, n_sensitive=5, n_robust=5, max_total=12):
    """Compute a shared set of representative classes across multiple model per-class CSVs.

    For each model, we pick the top `n_sensitive` classes (largest 100→25 drop) and the top
    `n_robust` classes (smallest variance across coverages). Then we take the union across
    models and limit to `max_total` classes to keep plots legible. This ensures all models
    within the same dataset plot the same class set for direct comparison.
    """
    shared = []
    for mdir in models_dirs:
        matches = list(Path(mdir).glob("*per_class*.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[0])
        # Use stride=8 for stability (consistent with other plots)
        df8 = df[df['stride'] == 8]
        pivot = df8.groupby(['class', 'coverage'], as_index=False)['accuracy'].mean().pivot(index='class', columns='coverage', values='accuracy')
        if 100 not in pivot.columns or 25 not in pivot.columns:
            continue
        drop_100_to_25 = (pivot[100] - pivot[25]).sort_values(ascending=False)
        sensitive = drop_100_to_25.head(n_sensitive).index.tolist()
        variance = pivot.var(axis=1).sort_values()
        robust = variance.head(n_robust).index.tolist()
        shared.extend(sensitive + robust)
    # Preserve order, unique, and limit total
    unique_shared = []
    for c in shared:
        if c not in unique_shared:
            unique_shared.append(c)
        if len(unique_shared) >= max_total:
            break
    return unique_shared


def generate_representative_plot(per_class_csv, output_dir, model_name, dataset_name=None, shared_classes=None):
    """Generate representative classes sensitivity analysis.

    If `shared_classes` is provided, use that list (same classes across models in dataset);
    otherwise, pick the top sensitive/robust classes for this model.
    """
    print("🎯 Generating representative classes plot...")

    df = pd.read_csv(per_class_csv)

    # Find best stride for the model (used previously to choose representative selection baseline)
    stride_performance = df.groupby('stride')['accuracy'].mean()
    best_stride = stride_performance.idxmax()
    print(f"Best stride: {best_stride} (mean accuracy: {stride_performance[best_stride]:.4f})")

    # Use best stride for computing per-class patterns
    df = df[df['stride'] == best_stride]

    # Aggregate
    df = df.groupby(['class', 'coverage'], as_index=False)['accuracy'].mean()
    pivot = df.pivot(index='class', columns='coverage', values='accuracy')

    coverages = sorted(df['coverage'].unique())

    # If shared_classes provided, use it; else compute top-5 sensitive and top-5 robust (fallback)
    if shared_classes is not None and len(shared_classes) > 0:
        classes = shared_classes
        print(f"Using shared classes for dataset {dataset_name}: {classes}")
    else:
        # Compute sensitive/robust classes for this model
        drop_100_to_25 = (pivot[100] - pivot[25]).sort_values(ascending=False)
        sensitive_classes = drop_100_to_25.head(5).index.tolist()
        variance = pivot.var(axis=1)
        robust_classes = variance.sort_values().head(5).index.tolist()
        classes = sensitive_classes + robust_classes
        print(f"Most sensitive classes: {sensitive_classes}")
        print(f"Most robust classes: {robust_classes}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Distinct palette and styles for clarity
    pal = sns.color_palette("tab10", n_colors=max(10, len(classes)))

    for i, cls in enumerate(classes):
        # Some classes may be missing for some models — skip gracefully
        if cls not in pivot.index:
            print(f"Warning: class {cls} not present in {per_class_csv}")
            continue

        # Decide label type based on whether this class is sensitive relative to 100->25 drop
        drop_val = (pivot.loc[cls, 100] - pivot.loc[cls, 25]) if 100 in pivot.columns and 25 in pivot.columns else 0
        is_sensitive = drop_val > 0
        label = f"{cls} ({'sensitive' if is_sensitive else 'robust'})"
        color = pal[i % len(pal)]
        style = '--' if is_sensitive else '-'

        # Convert to percentage for plotting
        accuracies = [pivot.loc[cls, cov] * 100.0 for cov in coverages]
        ax.plot(coverages, accuracies, label=label, color=color, linestyle=style, linewidth=2.2, marker='o', markersize=6)

    # Include dataset in title
    dataset_label = f" — {dataset_name.upper()}" if dataset_name else ""
    ax.set_title(f'Representative Classes: Sensitivity Analysis ({model_name.capitalize()}{dataset_label})', fontsize=16, fontweight='bold')

    ax.set_xlabel('Frame Coverage (%)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    # Format y-axis as percentage with one decimal
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    ax.set_xticks(coverages)
    ax.set_xticklabels([f"{int(c)}%" for c in coverages])
    ax.grid(True, alpha=0.25)

    # Place legend below as a compact multi-column legend to avoid overlapping the plot
    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.18), loc='upper center', fontsize=9)

    # No per-point annotations (clean lines as requested)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = output_dir / "per_class_representative.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_heatmap_plot(temporal_csv, output_dir, model_name):
    """Generate accuracy heatmap across coverage and stride."""
    print("🔥 Generating accuracy heatmap...")

    df = pd.read_csv(temporal_csv)

    # Pivot to create heatmap
    heatmap_data = df.pivot_table(values='accuracy', index='coverage', columns='stride', aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy'})

    plt.title(f'Accuracy Heatmap: Coverage vs Stride ({model_name.capitalize()})', fontsize=16, fontweight='bold')
    plt.xlabel('Stride', fontsize=14)
    plt.ylabel('Coverage (%)', fontsize=14)

    output_path = output_dir / "accuracy_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_accuracy_curves(temporal_csv, output_dir, model_name, dataset_name=None):
    """Generate accuracy vs coverage curves for different strides, including dataset in title."""
    print("📈 Generating accuracy curves...")

    df = pd.read_csv(temporal_csv)

    plt.figure(figsize=(12, 8))

    strides = sorted(df['stride'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(strides)))

    for i, stride in enumerate(strides):
        stride_data = df[df['stride'] == stride].sort_values('coverage')
        plt.plot(stride_data['coverage'], stride_data['accuracy'] * 100.0,
                label=f'Stride {stride}', color=colors[i], linewidth=2, marker='o')

    plt.xlabel('Frame Coverage (%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    # Format y-axis as percent with one decimal
    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    dataset_label = f" — {dataset_name.upper()}" if dataset_name else ""
    plt.title(f'Accuracy vs Coverage by Stride ({model_name.capitalize()}{dataset_label})', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = output_dir / "accuracy_vs_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def generate_coverage_curves_composite(evaluations_base_dir, output_dir):
    """Generate a 2×3 composite of accuracy vs coverage curves (rows=datasets, cols=models).

    Each subplot shows coverage (x-axis) vs accuracy (%) curves for all strides, with a compact legend.
    """
    print("🖼️ Generating coverage-degradation 2×3 composite...")

    datasets = ['ucf101', 'kinetics400']
    models = ['timesformer', 'videomae', 'vivit']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            model_dir = Path(evaluations_base_dir) / dataset / model
            matches = list(model_dir.glob("*temporal_sampling*.csv"))
            if not matches:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                continue

            temporal_csv = matches[0]
            df = pd.read_csv(temporal_csv)

            coverages = sorted(df['coverage'].unique())
            strides = sorted(df['stride'].unique())

            colors = plt.cm.viridis(np.linspace(0, 1, len(strides)))

            for k, stride in enumerate(strides):
                stride_data = df[df['stride'] == stride].sort_values('coverage')
                ax.plot(stride_data['coverage'], stride_data['accuracy'] * 100.0,
                        label=f'Stride {stride}', color=colors[k], linewidth=2, marker='o')

            ax.set_xlabel('Coverage (%)', fontsize=12)
            ax.set_title(f"{dataset.upper()} — {model.capitalize()}", fontsize=13)
            ax.set_xticks(coverages)
            ax.set_xticklabels([f"{int(c)}%" for c in coverages])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
            ax.grid(True, alpha=0.25)

            # Only show legend on bottom row center subplot to avoid clutter
            if i == 1 and j == 1:
                ax.legend(ncol=len(strides), bbox_to_anchor=(0.5, -0.35), loc='upper center', fontsize=10)

    fig.suptitle('Coverage Degradation Patterns Across Datasets and Architectures', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(output_dir) / 'coverage_degradation_composite.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved coverage-degradation composite: {out}")


def generate_distribution_composite(evaluations_base_dir, output_dir):
    """Generate a 2x3 composite figure: rows=datasets (UCF-101, Kinetics-400), cols=models (TimeSformer, VideoMAE, ViViT).

    This function expects the conventional folder structure under `evaluations/{dataset}/{model}` with per-class CSVs available.
    """
    print("🧩 Generating 2×3 composite distribution figure...")

    datasets = ['ucf101', 'kinetics400']
    models = ['timesformer', 'videomae', 'vivit']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')

    import matplotlib.ticker as mtick

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            model_dir = Path(evaluations_base_dir) / dataset / model
            # Find per_class CSV
            matches = list(model_dir.glob("*per_class*.csv"))
            if not matches:
                axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                axes[i, j].set_axis_off()
                continue

            per_class_csv = matches[0]
            df = pd.read_csv(per_class_csv)
            # Use stride 8 for the distributions (consistent across panels)
            df = df[df['stride'] == 8]

            coverages = sorted(df['coverage'].unique())
            class_list = sorted(df['class'].unique())

            acc_matrix = np.zeros((len(class_list), len(coverages)))
            for ii, cls in enumerate(class_list):
                for jj, cov in enumerate(coverages):
                    acc = df[(df['class'] == cls) & (df['coverage'] == cov)]['accuracy']
                    acc_matrix[ii, jj] = (acc.values[0] * 100.0) if not acc.empty else np.nan

            acc_df = pd.DataFrame(acc_matrix, columns=coverages)

            # Boxplot without fliers for clean compact display
            pal = sns.color_palette('viridis', n_colors=len(coverages))
            sns.boxplot(data=acc_df, ax=axes[i, j], palette=pal, fliersize=0, linewidth=1.0)

            # Annotate medians only
            medians = acc_df.median(axis=0)
            for k, m in enumerate(medians):
                axes[i, j].annotate(f"{m:.1f}%", xy=(k, m), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8, fontweight='semibold')

            axes[i, j].set_title(f"{dataset.upper()} — {model.capitalize()}", fontsize=12)
            axes[i, j].set_xticklabels([str(int(c)) for c in coverages], fontsize=10)
            axes[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
            axes[i, j].grid(True, linestyle=':', alpha=0.25)

    # Global layout adjustments
    fig.suptitle('Per-Class Accuracy Distributions (Stride = 8) — Comparative', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(output_dir) / 'per_class_distribution_composite.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved composite: {out}")


def generate_coverage_stride_composite(evaluations_base_dir, output_dir, cmap='viridis'):
    """Generate a 2x3 composite of coverage×stride heatmaps (rows=datasets, cols=models).

    Each cell displays mean accuracy (percent). Annotation color is chosen so white text highlights poorer performance (user preference), darker colors indicate higher accuracy.
    """
    print("🧭 Generating coverage×stride 2×3 composite...")

    datasets = ['ucf101', 'kinetics400']
    models = ['timesformer', 'videomae', 'vivit']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

    import matplotlib.ticker as mtick

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            model_dir = Path(evaluations_base_dir) / dataset / model
            # Find temporal CSV
            matches = list(model_dir.glob("*temporal_sampling*.csv"))
            if not matches:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                continue

            temporal_csv = matches[0]
            df = pd.read_csv(temporal_csv)

            # Pivot to create heatmap; ensure coverage rows are sorted descending (100 at top)
            heatmap_data = df.pivot_table(values='accuracy', index='coverage', columns='stride', aggfunc='mean') * 100.0
            heatmap_data = heatmap_data.sort_index(ascending=False)
            heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

            vmin = float(np.nanmin(heatmap_data.values))
            vmax = float(np.nanmax(heatmap_data.values))
            midpoint = (vmin + vmax) / 2.0

            sns.heatmap(heatmap_data, ax=ax, annot=False, fmt='.1f', cmap=cmap, cbar=False,
                        linewidths=0.3, linecolor='white', vmin=vmin, vmax=vmax)

            # Annotate each cell with percentage and user-preferred color mapping
            for y in range(heatmap_data.shape[0]):
                for x in range(heatmap_data.shape[1]):
                    val = heatmap_data.iloc[y, x]
                    if np.isnan(val):
                        txt = '—'
                    else:
                        txt = f"{val:.1f}%"
                    # Per user: use white text to indicate poorer performance (val < midpoint)
                    text_color = 'white' if (not np.isnan(val) and val < midpoint) else 'black'
                    ax.text(x + 0.5, y + 0.5, txt, ha='center', va='center', color=text_color, fontsize=9, fontweight='semibold')

            ax.set_title(f"{dataset.upper()} — {model.capitalize()}", fontsize=12)
            ax.set_xlabel('Stride', fontsize=11)
            ax.set_ylabel('Coverage (%)' if j == 0 else '', fontsize=11)
            ax.set_xticklabels([str(int(c)) for c in heatmap_data.columns], rotation=0)
            ax.set_yticklabels([str(int(c)) for c in heatmap_data.index], rotation=0)
            ax.grid(False)

    # Add a single colorbar on the right
    cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mean Accuracy (%)', fontsize=12)

    fig.suptitle('Coverage × Stride: Mean Accuracy (per dataset × model)', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])

    out = Path(output_dir) / 'coverage_stride_interactions.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved coverage×stride composite: {out}")

def main():
    parser = argparse.ArgumentParser(description="Generate all analysis plots for a model and dataset")
    parser.add_argument('--model', required=True, help='Model name (videomae, vivit, timesformer, or all)')
    parser.add_argument('--dataset', required=True, choices=['ucf101', 'kinetics400'], help='Dataset name')
    parser.add_argument('--stride', type=int, default=8, help='Stride for distribution analysis (default: 8)')

    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).parent.parent.parent
    evaluations_dir = base_dir / "evaluations" / args.dataset

    if args.model == 'all':
        models = ['videomae', 'vivit', 'timesformer']
    else:
        models = [args.model]

    for model in models:
        print(f"\n{'='*60}")
        print(f"🎨 Generating plots for {model.upper()} on {args.dataset.upper()}")
        print(f"{'='*60}")

        model_dir = evaluations_dir / model
        if not model_dir.exists():
            print(f"⚠️  No results found for {model} on {args.dataset}")
            continue

        # Find CSV files (with fallbacks for different filename patterns)
        if args.dataset == 'kinetics400':
            per_class_csv = model_dir / f"{model}-base-finetuned-kinetics_per_class.csv"
            temporal_csv = model_dir / f"{model}-base-finetuned-kinetics_temporal_sampling.csv"
        else:  # ucf101
            per_class_csv = model_dir / f"fine_tuned_{model}_{args.dataset}_per_class_testset.csv"
            temporal_csv = model_dir / f"fine_tuned_{model}_{args.dataset}_temporal_sampling.csv"

        # Fallback: search for any file containing 'per_class' or 'temporal_sampling'
        if not per_class_csv.exists():
            matches = list(model_dir.glob("*per_class*.csv"))
            per_class_csv = matches[0] if matches else per_class_csv

        if not temporal_csv.exists():
            matches = list(model_dir.glob("*temporal_sampling*.csv"))
            temporal_csv = matches[0] if matches else temporal_csv

        if not per_class_csv.exists():
            print(f"❌ Per-class CSV not found: {per_class_csv}")
            continue

        if not temporal_csv.exists():
            print(f"❌ Temporal CSV not found: {temporal_csv}")
            continue

        # Generate all plots
        try:
            # For the dataset-level shared class selection, we'll delay computing until we have seen all models
            generate_distribution_plot(per_class_csv, model_dir, model, args.stride)
            # Representative plot will be generated later once we compute shared classes for the dataset
            # For now, create the heatmap and accuracy curves and run stats
            generate_heatmap_plot(temporal_csv, model_dir, model)
            generate_accuracy_curves(temporal_csv, model_dir, model, args.dataset)

            # Run statistical analysis
            run_statistical_analysis(str(temporal_csv), str(per_class_csv), model_dir)

            print(f"✅ Basic plots generated for {model}")

        except Exception as e:
            print(f"❌ Error generating plots for {model}: {e}")

    # After all models for this dataset are processed, compute shared classes across models and regenerate representative plots to use the same classes
    models_dirs = [evaluations_dir / m for m in models]
    shared = compute_shared_classes(models_dirs, n_sensitive=5, n_robust=5, max_total=12)
    for model in models:
        model_dir = evaluations_dir / model
        matches = list(model_dir.glob("*per_class*.csv"))
        if not matches:
            continue
        generate_representative_plot(matches[0], model_dir, model, dataset_name=args.dataset, shared_classes=shared)

    print(f"✅ All plots generated for {args.dataset.upper()}")

    # When running all models for a dataset, generate the comparative 2×3 composite
    if args.model == 'all':
        comp_out = base_dir / 'evaluations' / 'comparative'
        comp_out.mkdir(parents=True, exist_ok=True)
        generate_distribution_composite(base_dir / 'evaluations', comp_out)
        # Also generate the coverage×stride composite heatmap
        generate_coverage_stride_composite(base_dir / 'evaluations', comp_out, cmap='viridis')
        # And generate the coverage-degradation curves composite (6 panels)
        generate_coverage_curves_composite(base_dir / 'evaluations', comp_out)

    print(f"\n🎉 Plot generation complete!")

if __name__ == "__main__":
    main()