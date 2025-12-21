import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_accuracy_curves(df_results: pd.DataFrame, output_dir: str = "UCF101_data/results"):
    """Plot accuracy curves and save to file."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for s in sorted(df_results["stride"].unique()):
        sub = df_results[df_results["stride"] == s]
        plt.plot(sub["coverage"], sub["accuracy"], marker="o", label=f"stride={s}")
    plt.xlabel("Frame Coverage (%)")
    plt.ylabel("Accuracy")
    plt.title("Temporal Sampling Effects")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(output_dir, "accuracy_vs_coverage.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


def plot_heatmap(df_results: pd.DataFrame, output_dir: str = "UCF101_data/results"):
    """Plot heatmap and save to file."""
    os.makedirs(output_dir, exist_ok=True)
    pivot = df_results.pivot(index="coverage", columns="stride", values="accuracy")
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
    plt.title("Accuracy Heatmap: Coverage vs Stride")
    plt.xlabel("Stride")
    plt.ylabel("Frame Coverage (%)")
    out_path = os.path.join(output_dir, "accuracy_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


def plot_per_class_lines(df_perclass: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_perclass, x="coverage", y="accuracy", hue="class", style="stride", legend=False, alpha=0.5)
    plt.title("Accuracy vs Frame Coverage per Class")
    plt.xlabel("Frame Coverage (%)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_aliasing_bar(aliasing_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    top_sensitive = aliasing_df.groupby("class")["drop_25_to_100"].mean().sort_values(ascending=False).head(15)
    top_sensitive.plot(kind="bar", color="tomato")
    plt.title("Top 15 Aliasing-Sensitive Classes")
    plt.ylabel("Accuracy Drop (100% → 25%)")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_class_stride_heatmap(df_perclass: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    pivot = df_perclass.groupby(["class", "stride"])["accuracy"].mean().unstack()
    sns.heatmap(pivot, cmap="viridis", linewidths=0.3)
    plt.title("Mean Accuracy per Class and Stride")
    plt.xlabel("Stride")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()


def boxplot_accuracy(df_perclass: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_perclass, x="class", y="accuracy", color="skyblue")
    plt.title("Distribution of Accuracy Across Sampling Configurations")
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
