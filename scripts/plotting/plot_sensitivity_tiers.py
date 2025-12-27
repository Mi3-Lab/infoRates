import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='Path to per-class CSV')
parser.add_argument('--stride', type=int, default=1, help='Stride to analyze')
args = parser.parse_args()

csv_path = args.csv
stride = args.stride
output_dir = '/'.join(csv_path.split('/')[:-1])

df = pd.read_csv(csv_path)
df = df[df['stride'] == stride]

pivot = df.pivot(index='class', columns='coverage', values='accuracy')

if 100 in pivot.columns and 25 in pivot.columns:
    drop = pivot[100] - pivot[25]
    drop = drop.dropna()

    # Categorize into tiers
    quantiles = drop.quantile([0.25, 0.5, 0.75])
    low = drop <= quantiles[0.25]
    medium = (drop > quantiles[0.25]) & (drop <= quantiles[0.75])
    high = drop > quantiles[0.75]

    counts = [low.sum(), medium.sum(), high.sum()]
    labels = ['Low Sensitivity', 'Medium Sensitivity', 'High Sensitivity']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'orange', 'red'], alpha=0.7)
    plt.ylabel('Number of Classes', fontsize=14)
    plt.title(f'Per-Class Sensitivity Tiers to Aliasing\n(Stride={stride})', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(counts):
        plt.text(i, v + 1, str(v), ha='center', fontsize=12)
    plt.tight_layout()

    output_path = f"{output_dir}/per_class_sensitivity_tiers.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

    print(f"Low: {counts[0]}, Medium: {counts[1]}, High: {counts[2]}")
else:
    print("Required coverages not found")