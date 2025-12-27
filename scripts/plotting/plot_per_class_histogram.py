import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='Path to per-class CSV')
parser.add_argument('--coverage', type=int, default=100, help='Coverage to analyze')
parser.add_argument('--stride', type=int, default=1, help='Stride to analyze')
args = parser.parse_args()

csv_path = args.csv
coverage = args.coverage
stride = args.stride
output_dir = '/'.join(csv_path.split('/')[:-1])

df = pd.read_csv(csv_path)

# Filter to specific coverage and stride
df_filtered = df[(df['coverage'] == coverage) & (df['stride'] == stride)]

if df_filtered.empty:
    print(f"No data for coverage={coverage}, stride={stride}")
    exit()

accuracies = df_filtered['accuracy'].values

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(accuracies, bins=20, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Per-Class Accuracy', fontsize=14)
plt.ylabel('Number of Classes', fontsize=14)
plt.title(f'Distribution of Per-Class Accuracies\n(Coverage={coverage}%, Stride={stride})', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = f"{output_dir}/per_class_accuracy_distribution.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved: {output_path}")

# Summary stats
print(f"Mean accuracy: {accuracies.mean():.4f}")
print(f"Std accuracy: {accuracies.std():.4f}")
print(f"Min: {accuracies.min():.4f}, Max: {accuracies.max():.4f}")