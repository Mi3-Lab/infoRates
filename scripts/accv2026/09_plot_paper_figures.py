#!/usr/bin/env python3
"""Generate paper figures for ACCV 2026.

Outputs (in evaluations/accv2026/paper_results/figures/):
  fig1_ssv2_budget_curves.pdf  — Acc vs frame budget for SSV2 (all models)
  fig2_cross_dataset_curves.pdf — Acc vs budget per dataset×model
  fig3_tds_auc_scatter.pdf     — TDS vs AUC scatter with correlation annotation
  fig4_critical_budget_bar.pdf — Critical frame budget per model×dataset
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "evaluations/accv2026/paper_results"
OUT = DATA / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────

COLORS = {
    "R3D-18":      "#4C72B0",
    "MC3-18":      "#DD8452",
    "R(2+1)D-18":  "#55A868",
    "TimeSformer": "#C44E52",
    "ViViT":       "#8172B3",
    "SlowFast":    "#937860",
    "VideoMAE":    "#DA8BC3",
}
MARKERS = {
    "R3D-18": "o", "MC3-18": "s", "R(2+1)D-18": "^",
    "TimeSformer": "D", "ViViT": "v", "SlowFast": "P", "VideoMAE": "*",
}
DATASET_COLORS = {"SSV2": "#4C72B0", "UCF101": "#55A868", "HMDB51": "#DD8452", "Diving48": "#C44E52"}

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


# ─── Figure 1: SSV2 budget curves ────────────────────────────────────────────

def fig1_ssv2_budget_curves():
    curves = pd.read_csv(DATA / "paper_fig_budget_curves.csv")
    ssv2 = curves[curves["dataset"] == "SSV2"].copy()
    if ssv2.empty:
        print("  [SKIP] fig1: no SSV2 data")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, grp in ssv2.groupby("model"):
        grp = grp.sort_values("budget")
        ax.plot(grp["budget"], grp["accuracy"] * 100,
                marker=MARKERS.get(model, "o"),
                color=COLORS.get(model, "gray"),
                label=model, linewidth=1.8, markersize=6)

    ax.set_xlabel("Frame Budget")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("SSV2: Accuracy vs Frame Budget")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = OUT / "fig1_ssv2_budget_curves.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Figure 2: Cross-dataset budget curves (one panel per model) ──────────────

def fig2_cross_dataset_curves():
    curves = pd.read_csv(DATA / "paper_fig_budget_curves.csv")
    multi_models = ["R(2+1)D-18", "SlowFast", "VideoMAE"]
    datasets_order = ["SSV2", "UCF101", "HMDB51", "Diving48"]

    available_models = [m for m in multi_models if m in curves["model"].unique()]
    if not available_models:
        print("  [SKIP] fig2: no multi-dataset data")
        return

    fig, axes = plt.subplots(1, len(available_models), figsize=(5 * len(available_models), 4.2), sharey=False)
    if len(available_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, available_models):
        model_df = curves[curves["model"] == model]
        for ds in datasets_order:
            ds_df = model_df[model_df["dataset"] == ds].sort_values("budget")
            if ds_df.empty:
                continue
            ax.plot(ds_df["budget"], ds_df["accuracy"] * 100,
                    marker="o", color=DATASET_COLORS.get(ds, "gray"),
                    label=ds, linewidth=1.8, markersize=5)
        ax.set_title(model)
        ax.set_xlabel("Frame Budget")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_xticks([4, 8, 16, 32])
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_ylim(bottom=0)

    fig.suptitle("Cross-Dataset: Accuracy vs Frame Budget", fontsize=13)
    fig.tight_layout()
    path = OUT / "fig2_cross_dataset_curves.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Figure 3: TDS vs AUC scatter ────────────────────────────────────────────

def fig3_tds_auc_scatter():
    tds_path = DATA / "paper_fig_tds_auc.csv"
    if not tds_path.exists():
        print("  [SKIP] fig3: TDS data not found")
        return

    df = pd.read_csv(tds_path)
    if df.empty:
        print("  [SKIP] fig3: TDS data empty")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    model_markers = {"R(2+1)D": "o", "VideoMAE": "s", "SlowFast": "^"}
    ds_color = {"SSV2": "#4C72B0", "UCF101": "#55A868", "HMDB51": "#DD8452", "Diving48": "#C44E52"}

    for _, row in df.iterrows():
        ax.scatter(row["tds_mean"], row["temporal_robustness_auc"],
                   color=ds_color.get(row["display"], "gray"),
                   marker=model_markers.get(row["model"], "o"),
                   s=90, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(f"{row['display']}\n({row['model']})",
                    (row["tds_mean"], row["temporal_robustness_auc"]),
                    textcoords="offset points", xytext=(6, 2), fontsize=7.5)

    # Fit line
    x = df["tds_mean"].values
    y = df["temporal_robustness_auc"].values
    if len(x) >= 3:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() * 0.95, x.max() * 1.05, 50)
        ax.plot(xs, m * xs + b, "k--", linewidth=1.2, alpha=0.6)

        from scipy import stats
        r, p = stats.spearmanr(x, -y)
        ax.text(0.97, 0.97, f"Spearman r = {r:.2f}\np = {p:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Legend for datasets
    for ds, c in ds_color.items():
        ax.scatter([], [], color=c, s=60, label=ds)
    ax.legend(title="Dataset", loc="lower left", framealpha=0.9)

    ax.set_xlabel("TDS (Mean FDE)")
    ax.set_ylabel("Temporal Robustness AUC")
    ax.set_title("Dataset Temporal Demand → Temporal Robustness")
    fig.tight_layout()
    path = OUT / "fig3_tds_auc_scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Figure 4: Critical frame budget bar chart ───────────────────────────────

def fig4_critical_budget_bar():
    t1_path = DATA / "paper_table1_ssv2.csv"
    t2_path = DATA / "paper_table2_multidataset.csv"

    rows = []
    if t1_path.exists():
        t1 = pd.read_csv(t1_path)
        for _, r in t1.iterrows():
            rows.append({"Model": r["Model"], "Dataset": "SSV2", "Critical_Budget": int(r["Critical_Budget"])})

    if t2_path.exists():
        t2 = pd.read_csv(t2_path)
        for _, r in t2.iterrows():
            rows.append({"Model": r["Model"], "Dataset": r["Dataset"], "Critical_Budget": int(r["Critical_Budget"])})

    if not rows:
        print("  [SKIP] fig4: no table data")
        return

    df = pd.DataFrame(rows)
    datasets = df["Dataset"].unique()
    models = df["Model"].unique()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            sub = df[(df["Model"] == model) & (df["Dataset"] == ds)]
            vals.append(int(sub["Critical_Budget"].iloc[0]) if not sub.empty else 0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model, color=COLORS.get(model, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Critical Frame Budget (frames)")
    ax.set_title("Critical Frame Budget by Model and Dataset")
    ax.set_yticks([4, 8, 16, 32])
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    path = OUT / "fig4_critical_budget_bar.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Figure 5: AUC heatmap (model × dataset) ─────────────────────────────────

def fig5_auc_heatmap():
    t2_path = DATA / "paper_table2_multidataset.csv"
    if not t2_path.exists():
        return

    t2 = pd.read_csv(t2_path)
    t2["AUC_float"] = t2["AUC"].astype(float)
    pivot = t2.pivot(index="Model", columns="Dataset", values="AUC_float")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if 0.4 < v < 0.85 else "white")
    plt.colorbar(im, ax=ax, label="AUC", fraction=0.046, pad=0.04)
    ax.set_title("Temporal Robustness AUC (Model × Dataset)")
    fig.tight_layout()
    path = OUT / "fig5_auc_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Figure 6: FDE routing vs fixed budget ───────────────────────────────────

def fig6_fde_routing():
    p = DATA / "fde_routing_ssv2_summary.csv"
    if not p.exists():
        print("  [SKIP] fig6: FDE routing summary not found")
        return

    df = pd.read_csv(p)
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fixed_acc = [float(v.strip("%")) for v in df["Fixed-16f Acc"]]
    adaptive_acc = [float(v.strip("%")) for v in df["Adaptive Acc"]]
    adaptive_frames = [float(v) for v in df["Adaptive Avg Frames"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy comparison
    ax = axes[0]
    ax.bar(x - width/2, fixed_acc, width, label="Fixed-16f", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, adaptive_acc, width, label="FDE Adaptive", color="#55A868", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy: FDE Adaptive vs Fixed-16f (SSV2)")
    ax.legend()
    ax.set_ylim(bottom=0)

    # Frame usage
    ax = axes[1]
    savings = [16 - f for f in adaptive_frames]
    colors = ["#55A868" if s > 0 else "#C44E52" for s in savings]
    ax.bar(x, adaptive_frames, color=colors, alpha=0.85)
    ax.axhline(16, color="black", linestyle="--", linewidth=1.5, label="Fixed-16f")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Average Frames Used")
    ax.set_title("FDE Adaptive Routing: Avg Frame Budget (SSV2)")
    ax.legend()
    for i, (frames, saving) in enumerate(zip(adaptive_frames, savings)):
        ax.text(i, frames + 0.3, f"{saving:+.1f}f", ha="center", fontsize=8,
                color="#55A868" if saving > 0 else "#C44E52")

    fig.tight_layout()
    path = OUT / "fig6_fde_routing_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def fig7_multiscale_fde():
    """Multi-scale: video, class, dataset FDE predictive power."""
    class_data_path = DATA / "class_fde_vs_auc_ssv2.csv"
    tds_path = DATA / "paper_fig_tds_auc.csv"
    eval_dir = ROOT / "evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_full_e10_a100"
    samples_path = eval_dir / "somethingv2_validation_accv2026_r2plus1d_18_ssv2_full_e10_a100_fixed_budget_samples.csv"
    fde_cache_path = eval_dir / "fde_cache.csv"

    if not (class_data_path.exists() and tds_path.exists() and samples_path.exists()):
        print("  [SKIP] fig7: missing data files")
        return

    from scipy import stats as _stats

    samples = pd.read_csv(samples_path)
    fde_cache = pd.read_csv(fde_cache_path)
    class_data = pd.read_csv(class_data_path)
    tds_df = pd.read_csv(tds_path)

    # Compute video-level optimal budget
    video_opt = {}
    for vid_id, grp in samples[~samples["skipped"].astype(bool)].groupby("video_id"):
        valid = grp[grp["correct_top1"].astype(bool)]
        video_opt[vid_id] = int(valid["budget"].min()) if not valid.empty else 32
    opt_df = pd.DataFrame(list(video_opt.items()), columns=["video_id", "optimal_budget"])
    merged_v = opt_df.merge(fde_cache, on="video_id")

    r_v, p_v = _stats.spearmanr(merged_v["fde"], merged_v["optimal_budget"])
    r_c, p_c = _stats.spearmanr(class_data["class_fde_mean"], -class_data["auc"])
    r_d, p_d = _stats.spearmanr(tds_df["tds_mean"], -tds_df["temporal_robustness_auc"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors_bud = {4: "#55A868", 8: "#4C72B0", 16: "#DD8452", 32: "#C44E52"}

    ax = axes[0]
    for b in [4, 8, 16, 32]:
        sub = merged_v[merged_v["optimal_budget"] == b]
        ax.scatter(sub["fde"], [b]*len(sub), alpha=0.05, s=3, color=colors_bud[b])
        if not sub.empty:
            ax.plot([sub["fde"].mean()], [b], "D", color=colors_bud[b], markersize=8, zorder=5, label=f"{b}f")
    ax.set_xlabel("FDE (per video)"); ax.set_ylabel("Optimal Budget")
    ax.set_title(f"Video-Level\nr={r_v:.3f}, p={p_v:.4f} (n={len(merged_v)})"); ax.set_yticks([4, 8, 16, 32])
    ax.legend(fontsize=8)

    ax = axes[1]
    for crit in sorted(class_data["critical_frame_budget"].unique()):
        sub = class_data[class_data["critical_frame_budget"] == crit]
        ax.scatter(sub["class_fde_mean"], sub["auc"], s=30, alpha=0.7, color=colors_bud.get(int(crit), "gray"), label=f"crit={crit}f")
    ax.set_xlabel("Class Mean FDE"); ax.set_ylabel("Class-Level AUC")
    ax.set_title(f"Class-Level\nr={r_c:.3f}, p={p_c:.4f} (n={len(class_data)}, NS)")
    ax.legend(fontsize=8)

    ax = axes[2]
    colors_ds = {"UCF101": "#55A868", "HMDB51": "#DD8452", "SSV2": "#4C72B0", "Diving48": "#C44E52"}
    for _, row in tds_df.iterrows():
        ax.scatter(row["tds_mean"], row["temporal_robustness_auc"],
                   color=colors_ds.get(row["display"], "gray"), s=120, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(f"{row['display']}", (row["tds_mean"], row["temporal_robustness_auc"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7.5)
    xs = np.linspace(tds_df["tds_mean"].min()-0.002, tds_df["tds_mean"].max()+0.002, 50)
    m, b_val = np.polyfit(tds_df["tds_mean"], tds_df["temporal_robustness_auc"], 1)
    ax.plot(xs, m*xs+b_val, "k--", alpha=0.6)
    ax.set_xlabel("TDS = Mean FDE (dataset)"); ax.set_ylabel("Temporal Robustness AUC")
    ax.set_title(f"Dataset-Level\nr={r_d:.3f}, p={p_d:.4f} (n={len(tds_df)})")

    fig.suptitle("FDE Predictive Power: Video → Class → Dataset Scale", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "fig7_multiscale_fde_analysis.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("Generating paper figures...")
    fig1_ssv2_budget_curves()
    fig2_cross_dataset_curves()
    fig3_tds_auc_scatter()
    fig4_critical_budget_bar()
    fig5_auc_heatmap()
    fig6_fde_routing()
    fig7_multiscale_fde()
    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
