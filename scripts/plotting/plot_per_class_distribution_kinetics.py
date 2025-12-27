import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='Path to per-class CSV')
parser.add_argument('--stride', type=int, default=8, help='Stride to analyze')
parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
args = parser.parse_args()

csv_path = args.csv
stride = args.stride
output_dir = args.output_dir or '/'.join(csv_path.split('/')[:-1])

df = pd.read_csv(csv_path)

# Only use specified stride
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

model_name = csv_path.split('/')[-3].capitalize()  # e.g., Videomae
plt.suptitle(f'Distribution of Per-Class Accuracies at Stride-{stride} Across Coverage Levels ({model_name})', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = f"{output_dir}/per_class_distribution_by_coverage.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved: {output_path}")