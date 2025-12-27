import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

try:
    import seaborn as sns
    sns.set(style="whitegrid", context="talk", font_scale=1.0)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='Path to per-class CSV')
parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
args = parser.parse_args()

csv_path = args.csv
output_dir = args.output_dir or '/'.join(csv_path.split('/')[:-1])

df = pd.read_csv(csv_path)

# Find the best stride (highest mean accuracy across all classes and coverages)
stride_performance = df.groupby('stride')['accuracy'].mean()
best_stride = stride_performance.idxmax()
print(f"Best stride: {best_stride} (mean accuracy: {stride_performance[best_stride]:.4f})")

# Use the best stride
df = df[df['stride'] == best_stride]

# Aggregate duplicates by taking mean accuracy
df = df.groupby(['class', 'coverage'], as_index=False)['accuracy'].mean()

# Pivot to get accuracy for each class at each coverage
pivot = df.pivot(index='class', columns='coverage', values='accuracy')

coverages = sorted(df['coverage'].unique())

# Compute metrics for sensitivity analysis
# Sensitive: largest drop from 100% to 25% coverage
drop_100_to_25 = pivot[100] - pivot[25]
sensitive_classes = drop_100_to_25.sort_values(ascending=False).head(5).index.tolist()

# Robust: lowest variance across all coverages (most consistent performance)
variance = pivot.var(axis=1)
robust_classes = variance.sort_values().head(5).index.tolist()

print(f"Most sensitive classes (largest 100%→25% drop): {sensitive_classes}")
print(f"Most robust classes (lowest variance across coverages): {robust_classes}")

# Use a better color palette
colors = plt.cm.Set1(np.linspace(0, 1, 10))
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, dashdotdot

plt.figure(figsize=(14, 10))

# Plot fragile (dashed, unique markers and colors)
for i, cls in enumerate(sensitive_classes):
    plt.plot(coverages, [pivot.loc[cls, c] for c in coverages],
             linestyle='--', color=colors[i], linewidth=2.5,
             marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2,
             label=f'{cls} (sensitive)', alpha=0.8)

# Plot robust (solid, unique markers and colors)
for i, cls in enumerate(robust_classes):
    plt.plot(coverages, [pivot.loc[cls, c] for c in coverages],
             linestyle='-', color=colors[i+5], linewidth=2.5,
             marker='s', markersize=8, markerfacecolor='white', markeredgewidth=2,
             label=f'{cls} (consistent)', alpha=0.8)

model_name = csv_path.split('/')[-1].split('_')[0]  # e.g., timesformer
plt.xlabel('Frame Coverage (%)', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
plt.title(f'{model_name}: Aliasing Sensitivity Analysis\nMost Sensitive vs Most Consistent Classes (stride={best_stride})', fontsize=20, fontweight='bold', pad=20)

# Legend in two columns, outside
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, frameon=True, ncol=2, fancybox=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(0, 1.05)
plt.xlim(min(coverages), max(coverages))
plt.xticks(coverages, [f"{int(c)}%" for c in coverages], fontsize=14)
plt.yticks(np.linspace(0, 1, 11), [f"{int(y*100)}" for y in np.linspace(0, 1, 11)], fontsize=14)

# Add some padding for the legend
plt.tight_layout(rect=[0, 0.05, 1, 1])

model_name = csv_path.split('/')[-3].capitalize()  # e.g., Videomae
plt.suptitle(f'Representative Classes: Sensitivity Analysis ({model_name})', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.96])

output_path = f"{output_dir}/per_class_representative.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_path}")