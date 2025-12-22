
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

try:
    import seaborn as sns
    sns.set(style="whitegrid", context="talk", font_scale=1.2)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Load per-class results
csv_path = "data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_per_class.csv"
df = pd.read_csv(csv_path)


# Use stride=8 to match TimeSformer plot
df = df[df['stride'] == 8]


# Pivot to get accuracy for each class at each coverage
pivot = df.pivot(index='class', columns='coverage', values='accuracy')

coverages = sorted(df['coverage'].unique())


# Compute drop from 100% to 25% coverage
drop = pivot[100] - pivot[25]


# Top 5 most sensitive classes (largest drop)
sensitive_classes = drop.sort_values(ascending=False).head(5).index.tolist()
# Top 5 most robust classes (smallest drop)
robust_classes = drop.sort_values().head(5).index.tolist()

coverages = sorted(df['coverage'].unique())


# Use tab10 color palette for up to 10 lines
import matplotlib.cm as cm
color_map = cm.get_cmap('tab10')
markers = ['o', 's', 'D', '^', 'P', 'X', 'v', '<', '>', '*']
fragile_markers = markers[:5]
robust_markers = markers[5:]

plt.figure(figsize=(12,8))

# Plot fragile (dashed, unique markers)
for i, cls in enumerate(sensitive_classes):
    plt.plot(coverages, [pivot.loc[cls, c] for c in coverages],
             linestyle='--', color=color_map(i), linewidth=2.5,
             marker=fragile_markers[i], markersize=10, label=cls)
# Plot robust (solid, unique markers)
for i, cls in enumerate(robust_classes):
    plt.plot(coverages, [pivot.loc[cls, c] for c in coverages],
             linestyle='-', color=color_map(i+5), linewidth=2.5,
             marker=robust_markers[i], markersize=10, label=cls)

plt.xlabel('Frame Coverage (%)', fontsize=18, fontweight='bold')
plt.ylabel('Accuracy', fontsize=18, fontweight='bold')
plt.title('Aliasing Sensitivity: Most Vulnerable (dashed) vs Robust (solid) Classes', fontsize=22, fontweight='bold', pad=20)

# Legend outside
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, frameon=True)
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(0, 1.05)
plt.xlim(min(coverages), max(coverages))
plt.xticks(coverages, [str(int(c)) for c in coverages], fontsize=14)
plt.yticks(np.linspace(0, 1, 11), [f"{int(y*100)}" for y in np.linspace(0, 1, 11)], fontsize=14)
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig("data/UCF101_data/results/videomae/per_class_representative.png", dpi=300)
plt.close()
print("Saved: data/UCF101_data/results/videomae/per_class_representative.png")
