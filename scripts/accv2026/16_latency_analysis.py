#!/usr/bin/env python3
"""Latency analysis: accuracy vs inference time Pareto frontier — ACCV 2026.

Reads all fixed_budget_summary.csv files, aggregates mean_inference_time_s,
and generates:
  - fig_latency_pareto.pdf        — per-dataset Pareto scatter (top1 vs time)
  - fig_latency_model_budget.pdf  — bar chart: inference time by model × budget
  - latency_summary.csv           — aggregated table
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "evaluations/accv2026/fixed_budget"
OUT    = ROOT / "evaluations/accv2026/paper_results/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "r3d18":       "R3D-18",
    "r3d_18":      "R3D-18",
    "mc3_18":      "MC3-18",
    "r2plus1d_18": "R2Plus1D-18",
    "slowfast_r50":"SlowFast-R50",
    "timesformer": "TimeSformer",
    "vivit":       "ViViT",
    "videomae":    "VideoMAE",
}
DS_MAP = {
    "ssv2":         "SSv2",
    "somethingv2":  "SSv2",
    "ucf101":       "UCF-101",
    "hmdb51":       "HMDB-51",
    "driveact":     "DriveAct",
    "diving48":     "Diving-48",
    "autsl":        "AUTSL",
    "epic_kitchens":"EPIC-Kitchens",
}

MODEL_ORDER = ["R3D-18", "MC3-18", "R2Plus1D-18", "SlowFast-R50", "TimeSformer", "ViViT", "VideoMAE"]
COLORS = {
    "R3D-18":       "#1f77b4",
    "MC3-18":       "#ff7f0e",
    "R2Plus1D-18":  "#2ca02c",
    "SlowFast-R50": "#d62728",
    "TimeSformer":  "#9467bd",
    "ViViT":        "#8c564b",
    "VideoMAE":     "#e377c2",
}
MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all() -> pd.DataFrame:
    records = []
    for f in sorted(IN_DIR.glob("*full*/*fixed_budget_summary.csv")):
        dir_name = f.parent.name.lower()
        model = next((v for k, v in MODEL_MAP.items() if k in dir_name), None)
        if model is None:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "dataset" in df.columns:
            ds_raw = str(df["dataset"].iloc[0]).lower()
        else:
            ds_raw = dir_name
        dataset = DS_MAP.get(ds_raw, ds_raw)
        for _, row in df.iterrows():
            records.append({
                "model":                model,
                "dataset":              dataset,
                "budget":               int(row["budget"]),
                "top1":                 float(row["top1"]) * 100,
                "mean_inference_time_s":float(row.get("mean_inference_time_s", 0)),
                "mean_total_time_s":    float(row.get("mean_total_time_s", 0)),
            })
    df = pd.DataFrame(records)
    # Keep only the most complete checkpoint per (model, dataset, budget)
    df = df.sort_values("top1").drop_duplicates(subset=["model", "dataset", "budget"], keep="last")
    return df.sort_values(["model", "dataset", "budget"])


# ---------------------------------------------------------------------------
# Figure 1: Pareto scatter — accuracy vs inference latency (aggregated over datasets)
# ---------------------------------------------------------------------------

def plot_pareto(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-point scatter (all model×dataset×budget combos)
    ax = axes[0]
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        for budget, grp in sub.groupby("budget"):
            ax.scatter(
                grp["mean_inference_time_s"] * 1000,  # ms
                grp["top1"],
                c=COLORS[model],
                marker=MARKERS.get(budget, "o"),
                s=60,
                alpha=0.75,
                label=f"{model} @ {budget}f" if False else None,
            )
    # Build legend for colors (models) and markers (budgets)
    model_patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODEL_ORDER if m in COLORS]
    budget_handles = [plt.scatter([], [], marker=MARKERS[b], c="gray", s=60, label=f"{b}f")
                      for b in [4, 8, 16, 32]]
    ax.set_xlabel("Mean inference time (ms/sample)", fontsize=12)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Inference Latency\n(all model×dataset×budget)", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    leg1 = ax.legend(handles=model_patches, loc="lower right", fontsize=8, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=budget_handles, loc="lower left", fontsize=8, title="Budget")

    # Right: averaged over datasets (model × budget)
    ax = axes[1]
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        grp_data = sub.groupby("budget")[["mean_inference_time_s", "top1"]].mean().reset_index()
        x = grp_data["mean_inference_time_s"] * 1000
        y = grp_data["top1"]
        ax.plot(x, y, "-o", color=COLORS[model], label=model, linewidth=1.5, markersize=7)
        for _, row in grp_data.iterrows():
            ax.annotate(f"{int(row['budget'])}f",
                        (row["mean_inference_time_s"] * 1000, row["top1"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color=COLORS[model])
    ax.set_xlabel("Mean inference time (ms/sample)", fontsize=12)
    ax.set_ylabel("Mean top-1 accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Latency (averaged across datasets)\nby model & budget", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = OUT / "fig_latency_pareto.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Inference time bar chart per model × budget
# ---------------------------------------------------------------------------

def plot_latency_bars(df: pd.DataFrame) -> None:
    budgets = sorted(df["budget"].unique())
    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    width = 0.18
    offsets = np.linspace(-(len(budgets)-1)/2, (len(budgets)-1)/2, len(budgets)) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, budget in enumerate(budgets):
        times = []
        for model in MODEL_ORDER:
            sub = df[(df["model"] == model) & (df["budget"] == budget)]
            times.append(sub["mean_inference_time_s"].mean() * 1000 if len(sub) > 0 else 0)
        ax.bar(x + offsets[i], times, width, label=f"{budget}f", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean inference time (ms/sample)", fontsize=11)
    ax.set_title("Model Inference Latency by Budget (averaged across datasets)", fontsize=12)
    ax.legend(title="Budget", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out_path = OUT / "fig_latency_model_budget.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Efficiency frontier — accuracy / latency tradeoff by model
# ---------------------------------------------------------------------------

def plot_efficiency(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].groupby("budget")[["mean_inference_time_s", "top1"]].mean().reset_index()
        # efficiency = accuracy per unit time (top1 / time_ms)
        sub["efficiency"] = sub["top1"] / (sub["mean_inference_time_s"] * 1000)
        ax.plot(sub["budget"], sub["efficiency"], "-o", color=COLORS[model], label=model, linewidth=2)
    ax.set_xlabel("Frame budget", fontsize=12)
    ax.set_ylabel("Efficiency: top-1% / ms  (higher = better)", fontsize=11)
    ax.set_title("Accuracy-Efficiency Tradeoff by Model & Budget\n(mean across datasets)", fontsize=12)
    ax.set_xticks([4, 8, 16, 32])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = OUT / "fig_latency_efficiency.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary(df: pd.DataFrame) -> None:
    agg = df.groupby(["model", "budget"]).agg(
        mean_top1=("top1", "mean"),
        mean_latency_ms=("mean_inference_time_s", lambda x: x.mean() * 1000),
        n_datasets=("dataset", "nunique"),
    ).reset_index()
    agg["top1_per_ms"] = agg["mean_top1"] / agg["mean_latency_ms"]
    out_path = ROOT / "evaluations/accv2026/paper_results/latency_summary.csv"
    agg.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved: {out_path}")
    print("\n=== Latency Summary (mean across datasets) ===")
    print(agg.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_all()
    print(f"Loaded {len(df)} data points: {df['model'].nunique()} models × "
          f"{df['dataset'].nunique()} datasets × {df['budget'].nunique()} budgets")
    if df.empty:
        print("[ERROR] No data found.")
        return
    plot_pareto(df)
    plot_latency_bars(df)
    plot_efficiency(df)
    save_summary(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
