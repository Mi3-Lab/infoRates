import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Filter to stride
df = df[df['stride'] == stride]

# Pivot
pivot = df.pivot(index='class', columns='coverage', values='accuracy')

# Compute drop from 100% to 25%
if 100 in pivot.columns and 25 in pivot.columns:
    drop = pivot[100] - pivot[25]
    drop = drop.dropna()

    # Plot histogram of drops
    plt.figure(figsize=(10, 6))
    sns.histplot(drop.values, bins=20, kde=True, color='salmon', edgecolor='black')
    plt.xlabel('Aliasing Drop (100% - 25% Coverage)', fontsize=14)
    plt.ylabel('Number of Classes', fontsize=14)
    plt.title(f'Distribution of Per-Class Aliasing Drops\n(Stride={stride})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{output_dir}/per_class_aliasing_drop.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

    # Save CSV of drops
    drop_df = drop.reset_index().rename(columns={0: 'aliasing_drop'})
    drop_csv_path = f"{output_dir}/per_class_aliasing_drop.csv"
    drop_df.to_csv(drop_csv_path, index=False)
    print(f"Saved: {drop_csv_path}")

    print(f"Mean drop: {drop.mean():.4f}")
    print(f"Std drop: {drop.std():.4f}")
else:
    print("100% or 25% coverage not found")