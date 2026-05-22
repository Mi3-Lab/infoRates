#!/usr/bin/env python3
"""
Gerar gráficos TRA Comparison no estilo do tra_comparison_timesformer.png
usando os dados fornecidos pelo usuário.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Seus dados (LaTeX table format -> estruturado)
data = {
    1: {25: (0.840, 0.851), 50: (0.860, 0.858), 75: (0.871, 0.866)},
    4: {25: (0.814, 0.842), 50: (0.843, 0.852), 75: (0.869, 0.863)},
    8: {25: (0.782, 0.821), 50: (0.830, 0.834), 75: (0.854, 0.850)},
    16: {25: (0.693, 0.723), 50: (0.716, 0.753), 75: (0.837, 0.839)},
}

# Converter para DataFrame (mesmo formato que o script original espera)
rows = []
for stride, cov_data in data.items():
    for coverage, (baseline, tra) in cov_data.items():
        rows.append({
            'coverage': coverage,
            'stride': stride,
            'accuracy_baseline': baseline,
            'accuracy_tra': tra,
        })

comparison_df = pd.DataFrame(rows)

# Calcular métricas
baseline_100_s1 = 0.871  # Baseline em 100% coverage, stride 1 (aproximado)
tra_100_s1 = 0.866  # TRA em 100% coverage, stride 1 (aproximado)

comparison_df['absolute_improvement'] = comparison_df['accuracy_tra'] - comparison_df['accuracy_baseline']
comparison_df['relative_improvement'] = (
    (comparison_df['accuracy_tra'] - comparison_df['accuracy_baseline']) / comparison_df['accuracy_baseline']
) * 100

print(comparison_df)
print()

# Preparar pivots para heatmaps
baseline_pivot = comparison_df.pivot(
    index='stride', columns='coverage', values='accuracy_baseline'
)
tra_pivot = comparison_df.pivot(
    index='stride', columns='coverage', values='accuracy_tra'
)
improvement_pivot = comparison_df.pivot(
    index='stride', columns='coverage', values='absolute_improvement'
)

print("Baseline Pivot:\n", baseline_pivot)
print("\nTRA Pivot:\n", tra_pivot)
print("\nImprovement Pivot:\n", improvement_pivot)

# Criar figura no MESMO estilo do tra_comparison_timesformer.png
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Temporal Robustness Augmentation (TRA) Comparison - TimeSformer', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Baseline heatmap
sns.heatmap(
    baseline_pivot,
    annot=True,
    fmt='.3f',
    cmap='YlOrRd_r',
    ax=axes[0, 0],
    cbar_kws={'label': 'Accuracy'},
    vmin=0.6, vmax=0.95,
    linewidths=0.5,
    linecolor='gray',
)
axes[0, 0].set_title('Baseline Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Coverage (%)', fontsize=11)
axes[0, 0].set_ylabel('Stride', fontsize=11)

# 2. TRA heatmap
sns.heatmap(
    tra_pivot,
    annot=True,
    fmt='.3f',
    cmap='YlOrRd_r',
    ax=axes[0, 1],
    cbar_kws={'label': 'Accuracy'},
    vmin=0.6, vmax=0.95,
    linewidths=0.5,
    linecolor='gray',
)
axes[0, 1].set_title('TRA Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Coverage (%)', fontsize=11)
axes[0, 1].set_ylabel('Stride', fontsize=11)

# 3. Improvement heatmap
sns.heatmap(
    improvement_pivot,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    ax=axes[1, 0],
    cbar_kws={'label': 'Improvement'},
    center=0,
    vmin=-0.15, vmax=0.15,
    linewidths=0.5,
    linecolor='gray',
)
axes[1, 0].set_title('Absolute Improvement (TRA - Baseline)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Coverage (%)', fontsize=11)
axes[1, 0].set_ylabel('Stride', fontsize=11)

# 4. Degradation curves (Baseline vs TRA for each stride across coverage)
colors = {1: '#e74c3c', 4: '#3498db', 8: '#2ecc71', 16: '#f39c12'}

for stride in sorted(comparison_df['stride'].unique()):
    subset = comparison_df[comparison_df['stride'] == stride].sort_values('coverage')
    
    # Baseline degradation
    axes[1, 1].plot(
        subset['coverage'],
        baseline_100_s1 - subset['accuracy_baseline'],
        marker='o',
        linestyle='--',
        linewidth=2,
        markersize=8,
        label=f'Baseline (S={stride})',
        color=colors[stride],
        alpha=0.6,
    )
    
    # TRA degradation
    axes[1, 1].plot(
        subset['coverage'],
        tra_100_s1 - subset['accuracy_tra'],
        marker='s',
        linestyle='-',
        linewidth=2.5,
        markersize=8,
        label=f'TRA (S={stride})',
        color=colors[stride],
        alpha=0.9,
    )

axes[1, 1].set_xlabel('Coverage (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy Degradation (from 100%/S1)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Degradation Curves: Baseline vs TRA', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=9, loc='upper left', ncol=2)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)

plt.tight_layout()
plt.savefig('docs/figures/tra_comparison_timesformer_new.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfico salvo em: docs/figures/tra_comparison_timesformer_new.png")

# Também salvar os JSONs no formato esperado
baseline_results = {}
tra_results = {}

for _, row in comparison_df.iterrows():
    key = f"cov{int(row['coverage'])}_stride{int(row['stride'])}"
    baseline_results[key] = float(row['accuracy_baseline'])
    tra_results[key] = float(row['accuracy_tra'])

Path("fine_tuned_models/tra_experiments/baseline").mkdir(parents=True, exist_ok=True)
Path("fine_tuned_models/tra_experiments/tra").mkdir(parents=True, exist_ok=True)

with open("fine_tuned_models/tra_experiments/baseline/robustness_timesformer.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

with open("fine_tuned_models/tra_experiments/tra/robustness_timesformer.json", "w") as f:
    json.dump(tra_results, f, indent=2)

print("✅ JSONs salvos em:")
print("   - fine_tuned_models/tra_experiments/baseline/robustness_timesformer.json")
print("   - fine_tuned_models/tra_experiments/tra/robustness_timesformer.json")
print()
print("="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Mean Absolute Improvement: {comparison_df['absolute_improvement'].mean():+.4f}")
print(f"Mean Relative Improvement: {comparison_df['relative_improvement'].mean():+.2f}%")
print(f"Best Improvement: {comparison_df['absolute_improvement'].max():+.4f}")
print(f"Worst Improvement: {comparison_df['absolute_improvement'].min():+.4f}")
print("="*80)

plt.show()
