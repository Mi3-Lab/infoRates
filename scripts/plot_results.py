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

    default_csv = Path(cfg.get("eval_out", "UCF101_data/results/ucf101_50f_finetuned.csv"))
    if default_csv.exists():
        df = pd.read_csv(default_csv)
        return df, default_csv

    # Fallback: pick latest CSV in results dir
    results_dir = Path(cfg.get("results_dir", "UCF101_data/results"))
    candidates = sorted(results_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {results_dir}")
    df = pd.read_csv(candidates[-1])
    return df, candidates[-1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_accuracy_vs_coverage(df: pd.DataFrame, out_dir: Path) -> Path:
    """Line plot: accuracy vs coverage for each stride."""
    plt.figure(figsize=(8, 5))
    for stride in sorted(df["stride"].unique()):
        subset = df[df["stride"] == stride].sort_values("coverage")
        plt.plot(subset["coverage"], subset["accuracy"], marker="o", label=f"stride={stride}")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Coverage by Stride")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / "accuracy_vs_coverage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_accuracy_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
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
    plt.title("Accuracy Heatmap (Stride vs Coverage)")
    plt.ylabel("Stride")
    plt.xlabel("Coverage (%)")
    out_path = out_dir / "accuracy_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_accuracy_per_time(df: pd.DataFrame, out_dir: Path) -> Path:
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
    plt.title("Efficiency: Accuracy/Time vs Coverage")
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


def plot_pareto_frontier(df: pd.DataFrame, out_dir: Path) -> Path:
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
    plt.title("Pareto-Optimal Frontier: Accuracy vs Latency")
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
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    csv_path = Path(args.csv) if args.csv else None

    df, source_csv = load_results(config_path, csv_path)

    # Output directory from config or CSV parent
    results_dir = None
    if config_path and config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        results_dir = Path(cfg.get("results_dir", "UCF101_data/results"))
    if results_dir is None:
        results_dir = source_csv.parent

    ensure_dir(results_dir)

    acc_cov = plot_accuracy_vs_coverage(df, results_dir)
    heatmap = plot_accuracy_heatmap(df, results_dir)
    eff_plot = plot_accuracy_per_time(df, results_dir)
    pareto_plot = plot_pareto_frontier(df, results_dir)

    summary = summarize(df)
    summary_md = write_summary_md(summary, results_dir, source_csv)

    # Save per-class aliasing summary if available
    per_class_csv = Path(args.per_class_csv) if args.per_class_csv else None
    if per_class_csv and per_class_csv.exists():
        df_per_class = pd.read_csv(per_class_csv)
        alias_summary = compute_per_class_aliasing_summary(df_per_class)
        if not alias_summary.empty:
            alias_path = results_dir / "per_class_aliasing_drop.csv"
            alias_summary.to_csv(alias_path, index=False)
            print(f"- {alias_path}")

    print("Saved:")
    print(f"- {acc_cov}")
    print(f"- {heatmap}")
    print(f"- {eff_plot}")
    print(f"- {pareto_plot}")
    print(f"- {summary_md}")


if __name__ == "__main__":
    main()
