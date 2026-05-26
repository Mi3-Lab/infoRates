#!/usr/bin/env python3
"""Figure: Unified routing method comparison for ACCV 2026 paper.

Reads cascade + knapsack global CSVs and plots accuracy vs. avg_frames for each method,
compared against fixed baselines. One subplot per dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "evaluations/accv2026/paper_results/figures"
CASCADE_GLOBAL = ROOT / "evaluations/accv2026/confidence_cascade/cascade_global_summary.csv"
KNAPSACK_GLOBAL = ROOT / "evaluations/accv2026/knapsack_confidence/knapsack_global_results.csv"
BUDGET_CURVES   = ROOT / "evaluations/accv2026/paper_results/paper_fig_budget_curves.csv"

DATASETS_ORDER = ["SSV2", "UCF101", "HMDB51", "Diving48"]
DATASET_LABELS = {"SSV2": "SSV2 (temporal)", "UCF101": "UCF101", "HMDB51": "HMDB51", "Diving48": "Diving48 (temporal)"}
# Maps canonical dataset name → possible substrings in each CSV's dataset column
DS_ALIASES = {
    "SSV2":    ["ssv2", "somethingv2", "something-something"],
    "UCF101":  ["ucf101", "ucf-101"],
    "HMDB51":  ["hmdb51", "hmdb-51"],
    "Diving48":["diving48", "diving-48"],
}

COLORS = {
    "fixed":            "#aaaaaa",
    "cascade_best":     "#e74c3c",
    "knapsack_learned": "#2ecc71",
    "oracle":           "#3498db",
}
MARKERS = {"fixed": "s", "cascade_best": "o", "knapsack_learned": "^", "oracle": "D"}


def load_fixed_baselines():
    """Load fixed-budget baselines per dataset (average over models)."""
    if not BUDGET_CURVES.exists():
        return {}
    df = pd.read_csv(BUDGET_CURVES)
    if "dataset" not in df.columns:
        return {}
    result = {}
    for ds in DATASETS_ORDER:
        sub = df[_ds_mask(df["dataset"], ds)]
        if sub.empty:
            continue
        acc_col = "accuracy" if "accuracy" in sub.columns else "top1"
        pivot = sub.groupby("budget")[acc_col].mean().to_dict()
        result[ds] = {int(b): float(v) for b, v in pivot.items()}
    return result


def _ds_mask(series: pd.Series, ds: str) -> pd.Series:
    """Return boolean mask matching a canonical dataset name via aliases."""
    aliases = DS_ALIASES.get(ds, [ds.lower()])
    mask = pd.Series(False, index=series.index)
    for a in aliases:
        mask |= series.str.lower().str.contains(a, na=False)
    return mask


def load_cascade_summary():
    """Load best cascade result per dataset (model-average, best k_low/k_high)."""
    if not CASCADE_GLOBAL.exists():
        return {}
    df = pd.read_csv(CASCADE_GLOBAL)
    result = {}
    for ds in DATASETS_ORDER:
        sub = df[_ds_mask(df["dataset"], ds)].copy()
        if sub.empty:
            continue
        rows = []
        for (kl, kh), g in sub.groupby(["k_low", "k_high"]):
            rows.append({
                "avg_frames": float(g["avg_frames"].mean()),
                "accuracy":   float(g["accuracy"].mean()),
            })
        pts = pd.DataFrame(rows).sort_values("avg_frames")
        result[ds] = pts
    return result


def load_knapsack_learned():
    """Load learned knapsack results per dataset (model-average)."""
    if not KNAPSACK_GLOBAL.exists():
        return {}
    df = pd.read_csv(KNAPSACK_GLOBAL)
    df = df[df["type"] == "knapsack_learned"].copy()
    result = {}
    for ds in DATASETS_ORDER:
        sub = df[_ds_mask(df["dataset"], ds)]
        if sub.empty:
            continue
        pts = sub.groupby("avg_frames").agg(accuracy=("accuracy", "mean")).reset_index()
        result[ds] = pts
    return result


def load_oracle_knapsack():
    """Load oracle knapsack per dataset (model-average)."""
    if not KNAPSACK_GLOBAL.exists():
        return {}
    df = pd.read_csv(KNAPSACK_GLOBAL)
    df = df[df["type"] == "oracle_knapsack"].copy()
    result = {}
    for ds in DATASETS_ORDER:
        sub = df[_ds_mask(df["dataset"], ds)]
        if sub.empty:
            continue
        pts = sub.groupby("avg_frames").agg(accuracy=("accuracy", "mean")).reset_index()
        result[ds] = pts
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed    = load_fixed_baselines()
    cascade  = load_cascade_summary()
    learned  = load_knapsack_learned()
    oracle   = load_oracle_knapsack()

    n_ds = len(DATASETS_ORDER)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4.2), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, DATASETS_ORDER):
        label = DATASET_LABELS.get(ds, ds)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Avg frames / video", fontsize=9)
        ax.set_ylabel("Top-1 accuracy (%)", fontsize=9) if ax == axes[0] else None

        # Fixed baselines
        if ds in fixed:
            xs = sorted(fixed[ds].keys())
            ys = [fixed[ds][b] * 100 for b in xs]
            ax.plot(xs, ys, linestyle="--", marker="s",
                    color=COLORS["fixed"], markersize=6, linewidth=1.5,
                    label="Fixed budget", zorder=2)
            # Add budget labels
            for x, y in zip(xs, ys):
                ax.annotate(f"{int(x)}f", (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7, color="#777777")

        # Cascade (best per avg_frames bin)
        if ds in cascade:
            pts = cascade[ds]
            ax.scatter(pts["avg_frames"], pts["accuracy"] * 100,
                       color=COLORS["cascade_best"], marker="o", s=40, zorder=4,
                       label="Cascade (confidence)", alpha=0.8)

        # Learned knapsack
        if ds in learned:
            pts = learned[ds].sort_values("avg_frames")
            ax.plot(pts["avg_frames"], pts["accuracy"] * 100,
                    color=COLORS["knapsack_learned"], marker="^", markersize=7,
                    linewidth=2, label="Knapsack-learned", zorder=5)

        # Oracle knapsack
        if ds in oracle:
            pts = oracle[ds].sort_values("avg_frames")
            ax.plot(pts["avg_frames"], pts["accuracy"] * 100,
                    color=COLORS["oracle"], marker="D", markersize=5,
                    linewidth=1.5, linestyle=":", label="Oracle knapsack", zorder=3)

        ax.set_xlim(2, 34)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        if ax == axes[0]:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Adaptive Frame Routing vs. Fixed Budget — ACCV 2026",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = OUT_DIR / "fig8_routing_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
