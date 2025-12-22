import argparse
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional seaborn for nicer heatmaps
try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


def load_results(config_path: Path | None, csv_path: Path | None) -> tuple[pd.DataFrame, Path]:
    """Load the evaluation CSV using config.yaml or explicit path."""
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        return df, csv_path

    if config_path is None:
        raise ValueError("Provide --config or --csv to locate results")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    default_csv = Path(cfg.get("eval_out", "data/UCF101_data/results/ucf101_50f_finetuned.csv"))
    if default_csv.exists():
        df = pd.read_csv(default_csv)
        return df, default_csv

    # Fallback: pick latest CSV in results dir
    results_dir = Path(cfg.get("results_dir", "data/UCF101_data/results"))
    candidates = sorted(results_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {results_dir}")
    df = pd.read_csv(candidates[-1])
    return df, candidates[-1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_accuracy_vs_coverage(df: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path:
    """Line plot: accuracy vs coverage for each stride."""
    plt.figure(figsize=(8, 5))
    for stride in sorted(df["stride"].unique()):
        subset = df[df["stride"] == stride].sort_values("coverage")
        plt.plot(subset["coverage"], subset["accuracy"], marker="o", label=f"stride={stride}")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy")
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Accuracy vs Coverage by Stride")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "accuracy_vs_coverage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_accuracy_heatmap(df: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path:
    """Heatmap of accuracy by stride x coverage."""
    pivot = df.pivot(index="stride", columns="coverage", values="accuracy")
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "Accuracy"})
    else:
        # Minimal heatmap using imshow when seaborn is unavailable
        import numpy as np
        data = pivot.values
        im = plt.imshow(data, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Accuracy")
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(s) for s in pivot.index])
        ax = plt.gca()
        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Accuracy Heatmap (Stride vs Coverage)")
    plt.ylabel("Stride")
    plt.xlabel("Coverage (%)")
    out_path = out_dir / "accuracy_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_accuracy_per_time(df: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path:
    """Efficiency plot: accuracy per second vs coverage (per stride)."""
    # Avoid division by zero
    df_eff = df.copy()
    df_eff["acc_per_sec"] = df_eff["accuracy"] / df_eff["avg_time"].replace(0, pd.NA)
    plt.figure(figsize=(8, 5))
    for stride in sorted(df_eff["stride"].unique()):
        subset = df_eff[df_eff["stride"] == stride].sort_values("coverage")
        plt.plot(subset["coverage"], subset["acc_per_sec"], marker="o", label=f"stride={stride}")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy per second")
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Efficiency: Accuracy/Time vs Coverage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "accuracy_per_second.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Pareto-optimal (accuracy, latency) frontier.
    Points where you cannot improve accuracy without worsening latency (or vice versa).
    """
    df_sorted = df.sort_values("avg_time")
    frontier = []
    best_acc = -1.0
    for _, row in df_sorted.iterrows():
        acc = row["accuracy"]
        if acc >= best_acc:
            frontier.append(row)
            best_acc = acc
    return pd.DataFrame(frontier)


def plot_pareto_frontier(df: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path:
    """Plot Pareto frontier: accuracy vs latency."""
    frontier = compute_pareto_frontier(df)
    plt.figure(figsize=(10, 6))
    # Plot all points
    plt.scatter(df["avg_time"], df["accuracy"], alpha=0.5, s=50, label="All configs", color="lightblue")
    # Highlight frontier
    plt.scatter(frontier["avg_time"], frontier["accuracy"], alpha=0.9, s=150, label="Pareto frontier", color="red", marker="*")
    # Annotate frontier points
    for _, row in frontier.iterrows():
        plt.annotate(
            f"c{int(row['coverage'])}s{int(row['stride'])}",
            (row["avg_time"], row["accuracy"]),
            fontsize=8,
            alpha=0.7
        )
    plt.xlabel("Avg Time per Sample (s)")
    plt.ylabel("Accuracy")
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Pareto-Optimal Frontier: Accuracy vs Latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "pareto_frontier.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def compute_per_class_aliasing_summary(df_per_class: pd.DataFrame) -> pd.DataFrame:
    """Compute per-class aliasing sensitivity: drop from 100% to 25% coverage.
    Returns DataFrame with class, mean_accuracy_100, mean_accuracy_25, and aliasing_drop.
    """
    if "coverage" not in df_per_class.columns or "accuracy" not in df_per_class.columns:
        return pd.DataFrame()
    
    pivot = (
        df_per_class[df_per_class["coverage"].isin([25, 100])]
        .groupby(["class", "coverage"], as_index=False)["accuracy"]
        .mean()
        .pivot(index="class", columns="coverage", values="accuracy")
    )
    if 25 not in pivot.columns or 100 not in pivot.columns:
        return pd.DataFrame()
    
    result = pivot.reset_index().rename(columns={100: "acc_100pct", 25: "acc_25pct"})
    result["aliasing_drop"] = result["acc_100pct"] - result["acc_25pct"]
    return result.sort_values("aliasing_drop", ascending=False)


def plot_per_class_aliasing_bar(df_per_class: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path | None:
    """Plot top-15 most aliasing-sensitive classes (100% → 25% drop)."""
    alias_summary = compute_per_class_aliasing_summary(df_per_class)
    if alias_summary.empty:
        return None
    top = alias_summary.head(15).set_index("class")
    plt.figure(figsize=(10, 6))
    top["aliasing_drop"].plot(kind="bar", color="tomato")
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Top 15 Aliasing-Sensitive Classes (Accuracy Drop 100% → 25%)")
    plt.ylabel("Accuracy Drop")
    plt.xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    out_path = out_dir / "per_class_aliasing_drop.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_per_class_stride_heatmap(df_per_class: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path | None:
    """Heatmap of mean accuracy per class × stride."""
    if "class" not in df_per_class.columns or "stride" not in df_per_class.columns:
        return None
    pivot = df_per_class.groupby(["class", "stride"])['accuracy'].mean().unstack()
    plt.figure(figsize=(12, 8))
    if HAS_SEABORN:
        sns.heatmap(pivot, cmap="viridis", linewidths=0.3)
    else:
        data = pivot.values
        im = plt.imshow(data, cmap="viridis", aspect="auto")
        plt.colorbar(im, label="Accuracy")
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(s) for s in pivot.index])
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Mean Accuracy per Class and Stride")
    plt.xlabel("Stride")
    plt.ylabel("Class")
    out_path = out_dir / "per_class_stride_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_per_class_accuracy_boxplot(df_per_class: pd.DataFrame, out_dir: Path, model_name: str = "") -> Path | None:
    """Boxplot of accuracy distribution across sampling configs per class (top-N classes by samples)."""
    if "class" not in df_per_class.columns or "accuracy" not in df_per_class.columns:
        return None
    # Select top 30 classes by sample count to keep figure readable
    counts = df_per_class.groupby("class")["n_samples"].sum().sort_values(ascending=False)
    top_classes = counts.head(30).index
    df_top = df_per_class[df_per_class["class"].isin(top_classes)]
    plt.figure(figsize=(12, 6))
    if HAS_SEABORN:
        sns.boxplot(data=df_top, x="class", y="accuracy", color="skyblue")
    else:
        # Fallback simple boxplot
        data = [df_top[df_top["class"] == c]["accuracy"].values for c in top_classes]
        plt.boxplot(data)
        plt.xticks(range(1, len(top_classes) + 1), top_classes)
    title_prefix = f"{model_name} " if model_name else ""
    plt.title(f"{title_prefix}Accuracy Distribution Across Sampling Configurations (Top-30 Classes)")
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    out_path = out_dir / "per_class_accuracy_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def summarize(df: pd.DataFrame) -> dict:
    """Compute key summary stats for paper."""
    best_idx = df["accuracy"].idxmax()
    best = df.loc[best_idx]
    by_stride = (
        df.sort_values(["stride", "accuracy"], ascending=[True, False])
        .groupby("stride", as_index=False)
        .first()[["stride", "coverage", "accuracy"]]
    )
    # Efficiency best (accuracy per second)
    df_eff = df.copy()
    df_eff["acc_per_sec"] = df_eff["accuracy"] / df_eff["avg_time"].replace(0, pd.NA)
    eff_idx = df_eff["acc_per_sec"].idxmax()
    eff_best = df_eff.loc[eff_idx]
    
    # Pareto frontier
    frontier = compute_pareto_frontier(df)
    
    return {
        "best_overall": {
            "coverage": int(best["coverage"]),
            "stride": int(best["stride"]),
            "accuracy": float(best["accuracy"]),
            "avg_time": float(best["avg_time"]),
        },
        "best_per_stride": by_stride.to_dict(orient="records"),
        "best_efficiency": {
            "coverage": int(eff_best["coverage"]),
            "stride": int(eff_best["stride"]),
            "acc_per_sec": float(eff_best["acc_per_sec"]),
            "accuracy": float(eff_best["accuracy"]),
            "avg_time": float(eff_best["avg_time"]),
        },
        "pareto_frontier": frontier,
    }


def write_summary_md(summary: dict, out_dir: Path, source_csv: Path) -> Path:
    lines = []
    lines.append(f"# Temporal Sampling Results Summary\n")
    lines.append(f"Source CSV: {source_csv}\n\n")
    
    bo = summary["best_overall"]
    lines.append("## Best Overall Configuration\n")
    lines.append(
        f"- **Accuracy**: {bo['accuracy']:.4f}\n"
        f"- **Coverage**: {bo['coverage']}%\n"
        f"- **Stride**: {bo['stride']}\n"
        f"- **Avg Time/Sample**: {bo['avg_time']:.4f}s\n\n"
    )
    
    be = summary["best_efficiency"]
    lines.append("## Best Efficiency Configuration (Accuracy per Second)\n")
    lines.append(
        f"- **Accuracy/Sec**: {be['acc_per_sec']:.2f}\n"
        f"- **Accuracy**: {be['accuracy']:.4f}\n"
        f"- **Coverage**: {be['coverage']}%\n"
        f"- **Stride**: {be['stride']}\n"
        f"- **Avg Time/Sample**: {be['avg_time']:.4f}s\n\n"
    )
    
    lines.append("## Best Configuration per Stride\n")
    for row in summary["best_per_stride"]:
        lines.append(
            f"- **Stride {int(row['stride'])}**: coverage={int(row['coverage'])}% → accuracy={float(row['accuracy']):.4f}\n"
        )
    
    lines.append("\n## Pareto-Optimal Frontier (Accuracy vs Latency)\n")
    lines.append("Configurations where accuracy cannot be improved without increasing latency.\n\n")
    frontier = summary.get("pareto_frontier", pd.DataFrame())
    if not frontier.empty:
        lines.append("| Coverage | Stride | Accuracy | Avg Time (s) |\n")
        lines.append("|----------|--------|----------|______________|\n")
        for _, row in frontier.iterrows():
            lines.append(
                f"| {int(row['coverage'])}% | {int(row['stride'])} | {float(row['accuracy']):.4f} | {float(row['avg_time']):.4f} |\n"
            )
    
    out_path = out_dir / "results_summary.md"
    out_path.write_text("".join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results and write a summary")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--csv", type=str, default=None, help="Optional explicit path to results CSV")
    parser.add_argument("--per-class-csv", type=str, default=None, help="Optional path to per-class results CSV")
    parser.add_argument("--wandb", action="store_true", help="Log generated plots to Weights & Biases")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    csv_path = Path(args.csv) if args.csv else None

    df, source_csv = load_results(config_path, csv_path)

    # Output directory from config or CSV parent

    # Always use the parent directory of the main CSV as the output directory
    results_dir = source_csv.parent
    ensure_dir(results_dir)

    # Determine model name for plot titles
    model_name = ""
    if "videomae" in str(source_csv).lower():
        model_name = "VideoMAE"
    elif "timesformer" in str(source_csv).lower():
        model_name = "TimeSformer"

    acc_cov = plot_accuracy_vs_coverage(df, results_dir, model_name)
    heatmap = plot_accuracy_heatmap(df, results_dir, model_name)
    eff_plot = plot_accuracy_per_time(df, results_dir, model_name)
    pareto_plot = plot_pareto_frontier(df, results_dir, model_name)

    summary = summarize(df)
    summary_md = write_summary_md(summary, results_dir, source_csv)

    # Save per-class outputs if available
    per_class_csv = Path(args.per_class_csv) if args.per_class_csv else None
    per_class_alias_csv_path = None
    per_class_alias_png = per_class_heatmap_png = per_class_box_png = None
    if per_class_csv and per_class_csv.exists():
        df_per_class = pd.read_csv(per_class_csv)
        alias_summary = compute_per_class_aliasing_summary(df_per_class)
        if not alias_summary.empty:
            per_class_alias_csv_path = results_dir / "per_class_aliasing_drop.csv"
            alias_summary.to_csv(per_class_alias_csv_path, index=False)
        # Generate per-class charts
        per_class_alias_png = plot_per_class_aliasing_bar(df_per_class, results_dir, model_name)
        per_class_heatmap_png = plot_per_class_stride_heatmap(df_per_class, results_dir, model_name)
        per_class_box_png = plot_per_class_accuracy_boxplot(df_per_class, results_dir, model_name)

    print("Saved:")
    print(f"- {acc_cov}")
    print(f"- {heatmap}")
    print(f"- {eff_plot}")
    print(f"- {pareto_plot}")
    print(f"- {summary_md}")
    if per_class_alias_csv_path:
        print(f"- {per_class_alias_csv_path}")
    if per_class_alias_png:
        print(f"- {per_class_alias_png}")
    if per_class_heatmap_png:
        print(f"- {per_class_heatmap_png}")
    if per_class_box_png:
        print(f"- {per_class_box_png}")

    # Optional: log to W&B
    if args.wandb:
        try:
            import wandb  # type: ignore
            # Project name from config or default
            cfg = yaml.safe_load(config_path.read_text()) if config_path and config_path.exists() else {}
            project = cfg.get("wandb_project", "inforates-ucf101")
            run = wandb.init(project=project, job_type="plots", config={"source_csv": str(source_csv)})
            wandb.log({
                "accuracy_vs_coverage": wandb.Image(str(acc_cov)),
                "accuracy_heatmap": wandb.Image(str(heatmap)),
                "accuracy_per_second": wandb.Image(str(eff_plot)),
                "pareto_frontier": wandb.Image(str(pareto_plot)),
            })
            # Per-class aliasing table if exists
            per_class_csv = Path(args.per_class_csv) if args.per_class_csv else None
            if per_class_csv and per_class_csv.exists():
                df_per_class = pd.read_csv(per_class_csv)
                alias_summary = compute_per_class_aliasing_summary(df_per_class)
                if not alias_summary.empty:
                    wandb.log({"per_class_aliasing_drop": wandb.Table(dataframe=alias_summary.head(20))})
                # Log per-class charts if generated
                if per_class_alias_png:
                    wandb.log({"per_class_aliasing_drop_chart": wandb.Image(str(per_class_alias_png))})
                if per_class_heatmap_png:
                    wandb.log({"per_class_stride_heatmap": wandb.Image(str(per_class_heatmap_png))})
                if per_class_box_png:
                    wandb.log({"per_class_accuracy_distribution": wandb.Image(str(per_class_box_png))})
            wandb.finish()
        except Exception as e:
            print(f"W&B logging skipped: {e}")


if __name__ == "__main__":
    main()
