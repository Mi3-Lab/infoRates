import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='Path to per-class CSV')
parser.add_argument('--coverage', type=int, default=100, help='Coverage to analyze')
args = parser.parse_args()

csv_path = args.csv
coverage = args.coverage
output_dir = '/'.join(csv_path.split('/')[:-1])

df = pd.read_csv(csv_path)
df = df[df['coverage'] == coverage]

pivot = df.pivot(index='class', columns='stride', values='accuracy')

# Sort classes by mean accuracy
mean_acc = pivot.mean(axis=1).sort_values(ascending=False)
pivot = pivot.loc[mean_acc.index]

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.xlabel('Stride', fontsize=14)
plt.ylabel('Class', fontsize=14)
plt.title(f'Per-Class Accuracy Heatmap\n(Coverage={coverage}%)', fontsize=16)
plt.xticks(rotation=45)
plt.yticks([])  # Hide class names for space
plt.tight_layout()

output_path = f"{output_dir}/per_class_stride_heatmap.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved: {output_path}")