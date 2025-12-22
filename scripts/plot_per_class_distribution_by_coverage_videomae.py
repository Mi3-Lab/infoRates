import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load per-class results
csv_path = "data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_per_class.csv"
df = pd.read_csv(csv_path)

# Only use stride=8
stride = 8
df = df[df['stride'] == stride]

# Prepare data for plotting
coverages = sorted(df['coverage'].unique())
class_list = sorted(df['class'].unique())

# Create a matrix: rows=class, cols=coverage, values=accuracy
acc_matrix = np.zeros((len(class_list), len(coverages)))
for i, cls in enumerate(class_list):
    for j, cov in enumerate(coverages):
        acc = df[(df['class'] == cls) & (df['coverage'] == cov)]['accuracy']
        acc_matrix[i, j] = acc.values[0] if not acc.empty else np.nan

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Boxplot
sns.boxplot(data=pd.DataFrame(acc_matrix, columns=coverages), ax=axes[0])
axes[0].set_xlabel('Frame Coverage (%)', fontsize=14)
axes[0].set_ylabel('Per-Class Accuracy', fontsize=14)
axes[0].set_title('Boxplot: Per-Class Accuracy by Coverage', fontsize=16)
axes[0].set_xticklabels([str(int(c)) for c in coverages])
axes[0].grid(True, linestyle=':', alpha=0.5)

# Violin plot
sns.violinplot(data=pd.DataFrame(acc_matrix, columns=coverages), ax=axes[1], inner='quartile')
axes[1].set_xlabel('Frame Coverage (%)', fontsize=14)
axes[1].set_title('Violin: Per-Class Accuracy by Coverage', fontsize=16)
axes[1].set_xticklabels([str(int(c)) for c in coverages])
axes[1].grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Distribution of Per-Class Accuracies at Stride-8 Across Coverage Levels', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data/UCF101_data/results/videomae/per_class_distribution_by_coverage.png", dpi=300)
plt.close()
print("Saved: data/UCF101_data/results/videomae/per_class_distribution_by_coverage.png")
