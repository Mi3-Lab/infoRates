#!/usr/bin/env python3
"""Generate paper figures for ACCV 2026.

Outputs (evaluations/accv2026/paper_results/figures/):
  fig1_ssv2_budget_curves.{pdf,png}
  fig2_cross_dataset_curves.{pdf,png}
  fig3_tds_dataset_ranking.{pdf,png}
  fig4_critical_budget_bar.{pdf,png}
  fig5_auc_heatmap.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "evaluations/accv2026/paper_results"
OUT  = DATA / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "R3D-18":       "#4C72B0",
    "MC3-18":       "#DD8452",
    "R(2+1)D-18":   "#55A868",
    "SlowFast-R50": "#937860",
    "TimeSformer":  "#C44E52",
    "ViViT":        "#8172B3",
    "VideoMAE":     "#DA8BC3",
    "VideoMamba":   "#64B5CD",
}
MODEL_MARKERS = {
    "R3D-18": "o", "MC3-18": "s", "R(2+1)D-18": "^",
    "SlowFast-R50": "P", "TimeSformer": "D", "ViViT": "v",
    "VideoMAE": "*", "VideoMamba": "X",
}
DATASET_COLORS = {
    "SSv2": "#4C72B0", "UCF-101": "#55A868", "HMDB-51": "#DD8452",
    "Diving-48": "#C44E52", "AUTSL": "#8172B3",
    "DriveAct": "#937860", "EPIC-Kitchens": "#DA8BC3",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def save_fig(fig, path, dpi=300):
    """Save as PDF (LaTeX) and PNG (review). bbox_inches='tight' on both."""
    p = Path(path)
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {p.stem}.{{pdf,png}}")


# ─── Figure 1: All models on SSv2 ────────────────────────────────────────────

def fig1_ssv2_budget_curves():
    curves = pd.read_csv(DATA / "paper_fig_budget_curves.csv")
    ssv2 = curves[curves["dataset"] == "SSv2"].copy()
    if ssv2.empty:
        print("  [SKIP] fig1: no SSv2 data in paper_fig_budget_curves.csv")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, grp in ssv2.groupby("model"):
        grp = grp.sort_values("budget")
        ax.plot(grp["budget"], grp["accuracy"],
                marker=MODEL_MARKERS.get(model, "o"),
                color=MODEL_COLORS.get(model, "gray"),
                label=model, linewidth=1.8, markersize=6)

    ax.set_xlabel("Frame Budget")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("SSv2: Accuracy vs Frame Budget (all models)")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, OUT / "fig1_ssv2_budget_curves.pdf")
    plt.close(fig)


# ─── Figure 2: Cross-dataset curves (3 representative models) ────────────────

def fig2_cross_dataset_curves():
    curves = pd.read_csv(DATA / "paper_fig_budget_curves.csv")
    models = ["R(2+1)D-18", "SlowFast-R50", "VideoMAE"]
    datasets = ["SSv2", "UCF-101", "HMDB-51", "Diving-48", "AUTSL"]

    avail = [m for m in models if m in curves["model"].unique()]
    if not avail:
        print("  [SKIP] fig2: no cross-dataset data found")
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5.5 * len(avail), 4.5), sharey=False)
    if len(avail) == 1:
        axes = [axes]

    for ax, model in zip(axes, avail):
        mdf = curves[curves["model"] == model]
        plotted = False
        for ds in datasets:
            dsdf = mdf[mdf["dataset"] == ds].sort_values("budget")
            if dsdf.empty:
                continue
            ax.plot(dsdf["budget"], dsdf["accuracy"],
                    marker="o", color=DATASET_COLORS.get(ds, "gray"),
                    label=ds, linewidth=1.8, markersize=5)
            plotted = True
        if plotted:
            ax.set_xscale("log", base=2)
            ax.set_xticks([4, 8, 16, 32])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(model)
        ax.set_xlabel("Frame Budget")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_ylim(bottom=0)

    fig.suptitle("Cross-Dataset: Accuracy vs Frame Budget", fontsize=13)
    fig.tight_layout()
    save_fig(fig, OUT / "fig2_cross_dataset_curves.pdf")
    plt.close(fig)


# ─── Figure 3: Dataset TDS ranking (accuracy-based) ──────────────────────────

def fig3_tds_dataset_ranking():
    p = DATA / "paper_tds_by_dataset.csv"
    if not p.exists():
        print("  [SKIP] fig3: paper_tds_by_dataset.csv not found")
        return

    df = pd.read_csv(p, index_col=0).squeeze("columns")
    df = df.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [DATASET_COLORS.get(ds, "#888888") for ds in df.index]
    bars = ax.barh(df.index[::-1], df.values[::-1] * 100, color=colors[::-1], alpha=0.85)

    for bar, val in zip(bars, df.values[::-1] * 100):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"+{val:.1f}pp", va="center", fontsize=10)

    ax.set_xlabel("Mean Accuracy Gain: 4f → 32f (percentage points)")
    ax.set_title("Temporal Demand Score (TDS) by Dataset\n"
                 "Higher = more frames needed for best accuracy")
    ax.set_xlim(0, df.values.max() * 100 * 1.18)
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    save_fig(fig, OUT / "fig3_tds_dataset_ranking.pdf")
    plt.close(fig)


# ─── Figure 4: Critical frame budget per model×dataset ───────────────────────

def fig4_critical_budget_heatmap():
    p = DATA / "paper_table_tds_metrics.csv"
    if not p.exists():
        print("  [SKIP] fig4: paper_table_tds_metrics.csv not found")
        return

    df = pd.read_csv(p)
    pivot = df.pivot_table(index="Model", columns="Dataset",
                           values="Critical_Budget", aggfunc="first")

    dataset_order = ["UCF-101", "HMDB-51", "DriveAct", "SSv2", "EPIC-Kitchens", "Diving-48", "AUTSL"]
    model_order   = ["R3D-18", "MC3-18", "R(2+1)D-18", "SlowFast-R50",
                     "TimeSformer", "ViViT", "VideoMAE", "VideoMamba"]
    cols = [c for c in dataset_order if c in pivot.columns]
    rows = [r for r in model_order   if r in pivot.index]
    pivot = pivot.loc[rows, cols]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.3 + 1, len(rows) * 0.6 + 1.2))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn_r",
                   vmin=4, vmax=32, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = pivot.values[i, j]
            if not np.isnan(float(v)):
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if float(v) >= 24 else "black")
    plt.colorbar(im, ax=ax, label="Critical Frame Budget", fraction=0.03, pad=0.04,
                 ticks=[4, 8, 16, 32])
    ax.set_title("Critical Frame Budget (frames needed to reach 95% of best accuracy)")
    fig.tight_layout()
    save_fig(fig, OUT / "fig4_critical_budget_heatmap.pdf")
    plt.close(fig)


# ─── Figure 5: AUC heatmap ───────────────────────────────────────────────────

def fig5_auc_heatmap():
    p = DATA / "paper_table_tds_metrics.csv"
    if not p.exists():
        return

    df = pd.read_csv(p)
    pivot = df.pivot_table(index="Model", columns="Dataset",
                           values="Temporal_AUC", aggfunc="first")

    dataset_order = ["UCF-101", "HMDB-51", "DriveAct", "SSv2", "EPIC-Kitchens", "Diving-48", "AUTSL"]
    model_order   = ["R3D-18", "MC3-18", "R(2+1)D-18", "SlowFast-R50",
                     "TimeSformer", "ViViT", "VideoMAE", "VideoMamba"]
    cols = [c for c in dataset_order if c in pivot.columns]
    rows = [r for r in model_order   if r in pivot.index]
    pivot = pivot.loc[rows, cols]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.3 + 1, len(rows) * 0.6 + 1.2))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = float(pivot.values[i, j])
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.25 or v > 0.75 else "black")
    plt.colorbar(im, ax=ax, label="Temporal Robustness AUC", fraction=0.03, pad=0.04)
    ax.set_title("Temporal Robustness AUC\n(higher = accuracy stable across budgets)")
    fig.tight_layout()
    save_fig(fig, OUT / "fig5_auc_heatmap.pdf")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...")
    fig1_ssv2_budget_curves()
    fig2_cross_dataset_curves()
    fig3_tds_dataset_ranking()
    fig4_critical_budget_heatmap()
    fig5_auc_heatmap()
    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
