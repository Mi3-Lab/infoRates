#!/usr/bin/env python3
"""
Plot Temporal Robustness: Baseline vs TRA degradation across stride and coverage.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data estruturado
# Formato: stride -> coverage -> (baseline, tra)
data = {
    1: {
        25: (0.840, 0.851),
        50: (0.860, 0.858),
        75: (0.871, 0.866),
    },
    4: {
        25: (0.814, 0.842),
        50: (0.843, 0.852),
        75: (0.869, 0.863),
    },
    8: {
        25: (0.782, 0.821),
        50: (0.830, 0.834),
        75: (0.854, 0.850),
    },
    16: {
        25: (0.693, 0.723),
        50: (0.716, 0.753),
        75: (0.837, 0.839),
    },
}

strides = [1, 4, 8, 16]
coverages = [25, 50, 75]
colors_cov = {25: '#e74c3c', 50: '#3498db', 75: '#2ecc71'}
colors_model = {'baseline': 'o-', 'tra': 's--'}

# Figure com subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Temporal Robustness: Baseline vs TRA across Coverage Levels', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, coverage in enumerate(coverages):
    ax = axes[idx]
    
    base_accs = [data[s][coverage][0] for s in strides]
    tra_accs = [data[s][coverage][1] for s in strides]
    
    # Plot
    ax.plot(strides, base_accs, 'o-', linewidth=2.5, markersize=8, 
            label='Baseline', color='#2c3e50', zorder=3)
    ax.plot(strides, tra_accs, 's--', linewidth=2.5, markersize=8, 
            label='TRA', color='#27ae60', zorder=3)
    
    # Degradation bar (baseline - tra)
    degradations = np.array(base_accs) - np.array(tra_accs)
    colors_deg = ['#e74c3c' if d > 0 else '#27ae60' for d in degradations]
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.set_axisbelow(True)
    
    # Formatting
    ax.set_xlabel('Stride (temporal sampling rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Coverage = {coverage}%', fontsize=13, fontweight='bold')
    ax.set_xticks(strides)
    ax.set_xticklabels(strides)
    ax.set_ylim([0.65, 0.88])
    ax.legend(fontsize=11, loc='lower left')
    
    # Annotations: degradation values
    for i, (s, base, tra, deg) in enumerate(zip(strides, base_accs, tra_accs, degradations)):
        y_pos = max(base, tra) + 0.01
        improvement = tra - base
        sign = '+' if improvement >= 0 else ''
        ax.text(s, y_pos, f'{sign}{improvement:.2%}', ha='center', fontsize=9, 
                fontweight='bold', color=colors_deg[i])

plt.tight_layout()
plt.savefig('docs/figures/temporal_robustness_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo em: docs/figures/temporal_robustness_comparison.png")
plt.show()

# Figure 2: Degradation heatmap
fig, ax = plt.subplots(figsize=(10, 6))

# Preparar dados para heatmap
degradation_matrix = np.zeros((len(coverages), len(strides)))
for i, cov in enumerate(coverages):
    for j, stride in enumerate(strides):
        base, tra = data[stride][cov]
        degradation_matrix[i, j] = tra - base  # Positivo = melhoria, Negativo = piora

# Criar heatmap
im = ax.imshow(degradation_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)

# Configurar eixos
ax.set_xticks(range(len(strides)))
ax.set_yticks(range(len(coverages)))
ax.set_xticklabels(strides, fontsize=11)
ax.set_yticklabels([f'{c}%' for c in coverages], fontsize=11)
ax.set_xlabel('Stride', fontsize=12, fontweight='bold')
ax.set_ylabel('Coverage', fontsize=12, fontweight='bold')
ax.set_title('TRA Improvement over Baseline\n(Positive = TRA Better; Negative = Baseline Better)', 
             fontsize=13, fontweight='bold')

# Adicionar valores nas células
for i in range(len(coverages)):
    for j in range(len(strides)):
        value = degradation_matrix[i, j]
        text = ax.text(j, i, f'{value:+.1%}', ha='center', va='center',
                      color='white' if abs(value) > 0.075 else 'black',
                      fontsize=11, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Δ Accuracy (TRA - Baseline)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/figures/temporal_robustness_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Heatmap salvo em: docs/figures/temporal_robustness_heatmap.png")
plt.show()

# Figure 3: Absolute degradation vs Stride (all coverages together)
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(strides))
width = 0.25

for idx, cov in enumerate(coverages):
    base_accs = [data[s][cov][0] for s in strides]
    tra_accs = [data[s][cov][1] for s in strides]
    degradations = np.array(tra_accs) - np.array(base_accs)
    
    offset = (idx - 1) * width
    bars = ax.bar(x_pos + offset, degradations, width, label=f'{cov}% Coverage', 
                   color=colors_cov[cov], alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1%}', ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=9, fontweight='bold')

# Linha de referência (melhoria nula)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=2)

# Configurar eixos
ax.set_xlabel('Stride (temporal sampling rate)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Improvement (TRA - Baseline)', fontsize=12, fontweight='bold')
ax.set_title('TRA Improvement across Stride and Coverage Levels', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(strides)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--', zorder=1)
ax.set_axisbelow(True)
ax.set_ylim([-0.20, 0.15])

plt.tight_layout()
plt.savefig('docs/figures/temporal_robustness_degradation.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico de degradação salvo em: docs/figures/temporal_robustness_degradation.png")
plt.show()

print("\n" + "="*80)
print("📊 RESUMO DOS GRÁFICOS GERADOS")
print("="*80)
print()
print("✅ 1. temporal_robustness_comparison.png")
print("     - Baseline vs TRA accuracy para cada coverage level")
print("     - Visualiza o efeito do stride em cada configuração")
print()
print("✅ 2. temporal_robustness_heatmap.png")
print("     - Heatmap de melhoria TRA vs Baseline")
print("     - Verde = TRA melhor | Vermelho = Baseline melhor")
print()
print("✅ 3. temporal_robustness_degradation.png")
print("     - Bar chart comparando melhoria em todos os (coverage, stride)")
print("     - Positivo = TRA melhor | Negativo = Baseline melhor")
print()
print("="*80)
