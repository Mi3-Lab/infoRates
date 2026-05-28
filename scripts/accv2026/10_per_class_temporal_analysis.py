#!/usr/bin/env python3
"""Per-class temporal demand analysis for ACCV 2026.

For each action class, computes:
  - critical_frame_budget: min budget retaining 95% of class-level best accuracy
  - accuracy at each budget
  - class-level AUC

Outputs:
  evaluations/accv2026/fixed_budget/<eval_dir>/per_class_temporal_metrics.csv
  evaluations/accv2026/paper_results/figures/fig_per_class_temporal.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_fig(fig, path, dpi=300):
    """Save figure as both PDF (for LaTeX) and PNG (for review/slides)."""
    from pathlib import Path
    p = Path(path)
    save_fig(fig, p)
    save_fig(fig, p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL_BASE = ROOT / "evaluations/accv2026/fixed_budget"
FIGURE_OUT = ROOT / "evaluations/accv2026/paper_results/figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

# Mapping from label_id to class name for SSV2 (if available)
LABEL_NAMES: dict[int, str] = {}


def load_class_names(data_root: str | None) -> dict[int, str]:
    import json
    candidates = [
        Path(data_root) / "something-something-v2-labels.json" if data_root else None,
        Path(data_root) / "labels/labels.json" if data_root else None,
        ROOT / "data/Something_data/labels/labels.json",
    ]
    for p in candidates:
        if p and p.exists():
            with open(p) as f:
                data = json.load(f)
            if data and isinstance(next(iter(data.values())), str):
                return {int(v): k for k, v in data.items()}
    return {}


def compute_per_class_metrics(
    samples_df: pd.DataFrame,
    budgets: list[int],
    target_fraction: float = 0.95,
    min_videos: int = 3,
) -> pd.DataFrame:
    """Return per-class temporal metrics."""
    df = samples_df[~samples_df["skipped"].astype(bool)].copy()
    rows = []

    for label_id, grp in df.groupby("label_id"):
        class_budgets = sorted(grp["budget"].unique())
        acc_by_budget: dict[int, float] = {}
        for b in class_budgets:
            sub = grp[grp["budget"] == b]
            if len(sub) >= min_videos:
                acc_by_budget[int(b)] = float(sub["correct_top1"].mean())

        if len(acc_by_budget) < 2:
            continue

        accs = [acc_by_budget.get(b, float("nan")) for b in budgets]
        valid_accs = [a for a in accs if not np.isnan(a)]
        best_acc = max(valid_accs) if valid_accs else 0.0
        target = target_fraction * best_acc

        # Critical budget
        critical = max(budgets)
        for b in sorted(budgets):
            if acc_by_budget.get(b, 0.0) >= target:
                critical = b
                break

        # AUC
        xs = [b for b in budgets if b in acc_by_budget]
        ys = [acc_by_budget[b] for b in xs]
        if len(xs) >= 2:
            auc = float(np.trapz(ys, xs) / (max(xs) - min(xs)))
        else:
            auc = float(np.mean(ys)) if ys else 0.0

        n_videos = grp["video_id"].nunique() if "video_id" in grp.columns else len(grp[grp["budget"] == budgets[0]])

        rows.append({
            "label_id": int(label_id),
            "class_name": LABEL_NAMES.get(int(label_id), f"class_{label_id}"),
            "n_videos": n_videos,
            "critical_frame_budget": critical,
            "best_accuracy": best_acc,
            "auc": auc,
            **{f"acc@{b}f": acc_by_budget.get(b, float("nan")) for b in budgets},
        })

    return pd.DataFrame(rows).sort_values("critical_frame_budget")


def plot_per_class(df: pd.DataFrame, model_name: str, dataset: str, budgets: list[int], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1. Distribution of critical budgets
    ax = axes[0]
    crit_counts = df["critical_frame_budget"].value_counts().sort_index()
    bars = ax.bar([str(b) for b in sorted(crit_counts.index)], crit_counts.values,
                   color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"][:len(crit_counts)])
    ax.set_xlabel("Critical Frame Budget")
    ax.set_ylabel("Number of Classes")
    ax.set_title(f"Critical Budget Distribution\n{model_name} on {dataset}")
    for bar, val in zip(bars, crit_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9)

    # 2. AUC distribution
    ax = axes[1]
    ax.hist(df["auc"].dropna(), bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(df["auc"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={df['auc'].mean():.3f}")
    ax.set_xlabel("Class-Level AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"AUC Distribution\n{model_name} on {dataset}")
    ax.legend()

    # 3. Top/bottom classes by AUC
    ax = axes[2]
    top_n = 8
    top = df.nlargest(top_n, "auc")[["class_name", "auc"]].reset_index(drop=True)
    bot = df.nsmallest(top_n, "auc")[["class_name", "auc"]].reset_index(drop=True)
    combined = pd.concat([bot, top]).reset_index(drop=True)
    colors = ["#C44E52"] * top_n + ["#55A868"] * top_n
    y_pos = np.arange(len(combined))
    ax.barh(y_pos, combined["auc"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    labels = [n[:30] for n in combined["class_name"]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("AUC")
    ax.set_title(f"Lowest (red) & Highest (green) AUC Classes\n{model_name} on {dataset}")

    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

EVAL_RUNS = [
    ("R3D-18", "SSV2",
     "r3d18_ssv2_full_e10_a100",
     "somethingv2_validation_accv2026_r3d18_ssv2_full_e10_a100_fixed_budget_samples.csv"),
    ("MC3-18", "SSV2",
     "mc3_18_ssv2_full_e10_a100",
     "somethingv2_validation_accv2026_mc3_18_ssv2_full_e10_a100_fixed_budget_samples.csv"),
    ("R(2+1)D-18", "SSV2",
     "r2plus1d_18_ssv2_full_e10_a100",
     "somethingv2_validation_accv2026_r2plus1d_18_ssv2_full_e10_a100_fixed_budget_samples.csv"),
    ("TimeSformer", "SSV2",
     "timesformer_ssv2_full_e10_h200",
     "somethingv2_validation_accv2026_timesformer_ssv2_full_e10_h200_fixed_budget_samples.csv"),
    ("ViViT", "SSV2",
     "vivit_ssv2_full_e10_h200",
     "somethingv2_validation_accv2026_vivit_ssv2_full_e10_h200_fixed_budget_samples.csv"),
    ("SlowFast", "SSV2",
     "slowfast_r50_ssv2_full_e10_a100",
     "somethingv2_validation_accv2026_slowfast_r50_ssv2_full_e10_a100_fixed_budget_samples.csv"),
    ("VideoMAE", "SSV2",
     "videomae_ssv2_full_e5_h200",
     "somethingv2_validation_accv2026_videomae_ssv2_full_e5_h200_fixed_budget_samples.csv"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None, help="SSV2 root for class names")
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--target-fraction", type=float, default=0.95)
    parser.add_argument("--min-videos", type=int, default=3)
    args = parser.parse_args()

    global LABEL_NAMES
    LABEL_NAMES = load_class_names(args.data_root)
    if LABEL_NAMES:
        print(f"Loaded {len(LABEL_NAMES)} class names")

    for model, dataset, eval_dir, samples_file in EVAL_RUNS:
        samples_path = EVAL_BASE / eval_dir / samples_file
        if not samples_path.exists():
            print(f"[SKIP] {model}/{dataset}: {samples_path} not found")
            continue

        print(f"\nAnalyzing {model} on {dataset}...")
        df = pd.read_csv(samples_path)
        metrics = compute_per_class_metrics(df, args.budgets, args.target_fraction, args.min_videos)
        print(f"  {len(metrics)} classes analyzed")
        print(f"  Critical budget distribution:")
        print(metrics["critical_frame_budget"].value_counts().sort_index().to_string())
        print(f"  AUC: mean={metrics['auc'].mean():.3f}  std={metrics['auc'].std():.3f}")
        print(f"  Most temporally demanding classes (lowest AUC):")
        print(metrics.nsmallest(5, "auc")[["class_name", "critical_frame_budget", "auc"]].to_string(index=False))
        print(f"  Least temporally demanding (highest AUC):")
        print(metrics.nlargest(5, "auc")[["class_name", "critical_frame_budget", "auc"]].to_string(index=False))

        out_csv = EVAL_BASE / eval_dir / "per_class_temporal_metrics.csv"
        metrics.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        out_fig = FIGURE_OUT / f"fig_per_class_{model.lower().replace('(','').replace(')','').replace('/','').replace('+','plus').replace('-','_')}_{dataset.lower()}.pdf"
        plot_per_class(metrics, model, dataset, args.budgets, out_fig)

    # Cross-model agreement: classes that all models agree are temporally demanding
    _all_metrics: dict[str, pd.DataFrame] = {}
    for model, dataset, eval_dir, _ in EVAL_RUNS:
        if dataset != "SSV2":
            continue
        p = EVAL_BASE / eval_dir / "per_class_temporal_metrics.csv"
        if p.exists():
            _all_metrics[model] = pd.read_csv(p).set_index("label_id")

    if len(_all_metrics) >= 3:
        print("\n" + "="*70)
        print("CROSS-MODEL AGREEMENT: SSV2 CLASS-LEVEL TEMPORAL DEMAND")
        print("="*70)
        # Average AUC across models for each class
        auc_frames = [df["auc"].rename(m) for m, df in _all_metrics.items()]
        auc_matrix = pd.concat(auc_frames, axis=1).dropna()
        auc_matrix["mean_auc"] = auc_matrix.mean(axis=1)
        auc_matrix["std_auc"] = auc_matrix.std(axis=1)
        auc_matrix["class_name"] = auc_matrix.index.map(LABEL_NAMES)

        print("\nMost temporally demanding (consistently low AUC across all models):")
        print(auc_matrix.nsmallest(10, "mean_auc")[["class_name", "mean_auc", "std_auc"]].to_string())
        print("\nLeast temporally demanding (consistently high AUC across all models):")
        print(auc_matrix.nlargest(10, "mean_auc")[["class_name", "mean_auc", "std_auc"]].to_string())

        out_csv = ROOT / "evaluations/accv2026/paper_results/per_class_cross_model_ssv2.csv"
        auc_matrix.reset_index().to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

        # Figure: scatter of class mean_auc vs std_auc
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(auc_matrix["mean_auc"], auc_matrix["std_auc"],
                        c=auc_matrix["mean_auc"], cmap="RdYlGn", s=40, alpha=0.75, vmin=0, vmax=0.9)
        plt.colorbar(sc, ax=ax, label="Mean AUC (across models)")
        # Annotate extremes
        for _, row in auc_matrix.nsmallest(3, "mean_auc").iterrows():
            ax.annotate(row["class_name"][:35], (row["mean_auc"], row["std_auc"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        for _, row in auc_matrix.nlargest(3, "mean_auc").iterrows():
            ax.annotate(row["class_name"][:35], (row["mean_auc"], row["std_auc"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Mean AUC (cross-model average)")
        ax.set_ylabel("Std AUC (cross-model disagreement)")
        ax.set_title("SSV2 Class-Level Temporal Demand (all models)")
        fig.tight_layout()
        out_fig = FIGURE_OUT / "fig_cross_model_class_auc_ssv2.pdf"
        save_fig(fig, out_fig)
        plt.close(fig)
        print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
