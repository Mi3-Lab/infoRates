"""Generate all paper and supplementary figures for ACCV 2026.

Main paper (8 pages):
  E1 aliasing curves — stride sensitivity per model family
  Cross-architecture stride sensitivity heatmap
  Fig 3: TDS ranking bar chart + E3 spectral correlation scatter
  E7 entropy routing — accuracy vs avg_frames
  E6 spatial resolution curves (SSv2)

Supplementary:
  Full E1 coverage×stride heatmaps
  E2 Levene variance plots
  E4 ANOVA eta² bar chart
  E5 action taxonomy
  E7 routing curves (all models)
  E10 clip duration

Usage:
  python scripts/accv2026/generate_paper_figures.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_MAIN = Path("evaluations/accv2026/paper_figures/main")
OUT_SUPP = Path("evaluations/accv2026/paper_figures/supplementary")
OUT_MAIN.mkdir(parents=True, exist_ok=True)
OUT_SUPP.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color scheme per architecture family
COLORS = {
    "r3d_18":       "#E64B35",   # red — CNN-3D
    "mc3_18":       "#F39B7F",   # salmon — CNN-mix
    "r2plus1d_18":  "#FF7F50",   # coral — CNN-sep
    "slowfast_r50": "#C0392B",   # dark red — CNN-dual
    "timesformer":  "#4DBBD5",   # blue — TSF(div)
    "vivit":        "#8FBC8F",   # green — ViViT(fact)
    "videomae":     "#1F78B4",   # dark blue — VMAE
    "videomamba":   "#7B2D8B",   # purple — SSM
}
LABELS = {
    "r3d_18": "R3D-18 (CNN-3D)",
    "mc3_18": "MC3-18 (CNN-mix)",
    "r2plus1d_18": "R2+1D (CNN-sep)",
    "slowfast_r50": "SlowFast (CNN-dual)",
    "timesformer": "TimeSformer (div-attn)",
    "vivit": "ViViT (fact-attn)",
    "videomae": "VideoMAE (MAE)",
    "videomamba": "VideoMamba (SSM)",
}
MARKERS = {
    "r3d_18": "o", "mc3_18": "s", "r2plus1d_18": "^",
    "slowfast_r50": "D", "timesformer": "o", "vivit": "s",
    "videomae": "^", "videomamba": "*",
}

MODELS   = ["r3d_18","mc3_18","r2plus1d_18","slowfast_r50","timesformer","vivit","videomae","videomamba"]
DATASETS = ["ucf101","ssv2","hmdb51","diving48","autsl","driveact","epic_kitchens","finegym"]
DATASET_LABELS = {
    "ucf101": "UCF-101", "ssv2": "SSv2", "hmdb51": "HMDB-51",
    "diving48": "Diving-48", "autsl": "AUTSL", "driveact": "DriveAct",
    "epic_kitchens": "EPIC-Kitchens", "finegym": "FineGym",
}
STRIDES = [1, 2, 4, 8, 16]
BASE = Path("evaluations/accv2026/coverage_stride_sweep")
# dashboard/data/sweep_summary.csv is the authoritative source (matches paper Table 2)
_DASHBOARD_CSV = Path("dashboard/data/sweep_summary.csv")
_DASHBOARD = None  # loaded on first use

def _get_dashboard():
    global _DASHBOARD
    if _DASHBOARD is None and _DASHBOARD_CSV.exists():
        _DASHBOARD = pd.read_csv(_DASHBOARD_CSV)
    return _DASHBOARD

NATIVE_RES = {
    "r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
    "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224,
}

def load_stride_curve(model, dataset, coverage=100):
    """Load accuracy vs stride at given coverage level.

    Priority:
    1. dashboard/data/sweep_summary.csv  (authoritative, matches paper tables)
    2. coverage_stride_sweep/{model}_{dataset}/sweep_summary.csv  (bare dir)
    3. coverage_stride_sweep/{model}_{dataset}_trainres{native}/  (trainres fallback)
    """
    dash = _get_dashboard()
    if dash is not None:
        sub = dash[
            (dash["model"] == model) &
            (dash["dataset"] == dataset) &
            (dash["coverage"] == coverage)
        ].sort_values("stride")
        if not sub.empty and 1 in sub["stride"].values and 16 in sub["stride"].values:
            return sub[["stride","top1"]].set_index("stride")["top1"]

    # fallback: bare dir (original E1 experiment)
    csv = BASE / f"{model}_{dataset}" / "sweep_summary.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        sub = df[df["coverage"] == coverage].sort_values("stride")
        if not sub.empty:
            return sub[["stride","top1"]].set_index("stride")["top1"]

    # trainres fallback: native-resolution fine-tuned checkpoint (e.g. videomamba_autsl_trainres224)
    native = NATIVE_RES.get(model, 224)
    csv_tr = BASE / f"{model}_{dataset}_trainres{native}" / "sweep_summary.csv"
    if csv_tr.exists():
        df = pd.read_csv(csv_tr)
        sub = df[df["coverage"] == coverage].sort_values("stride")
        if not sub.empty:
            return sub[["stride","top1"]].set_index("stride")["top1"]

    return None

def savefig(fig, name, folder=OUT_MAIN):
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {folder}/{name}.pdf/png")

# ── FIG 1: Aliasing curves — 3 representative datasets ───────────────────
print("Generating Fig 1: E1 aliasing curves...")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
show_datasets = ["autsl", "ssv2", "ucf101"]
tds_labels = {"autsl": "AUTSL (TDS#2, +53.0pp)", "ssv2": "SSv2 (TDS#3, +27.6pp)", "ucf101": "UCF-101 (TDS#8, +4.9pp)"}

for ax, ds in zip(axes, show_datasets):
    for model in MODELS:
        curve = load_stride_curve(model, ds, coverage=100)
        if curve is None:
            continue
        y = [curve.get(s, np.nan) * 100 for s in STRIDES]
        if all(np.isnan(v) for v in y):
            continue
        ax.plot(STRIDES, y, marker=MARKERS[model], color=COLORS[model],
                label=LABELS[model], lw=1.8, ms=5, alpha=0.9)

    ax.set_xscale("log", base=2)
    ax.set_xticks(STRIDES)
    ax.set_xticklabels([f"s={s}" for s in STRIDES], fontsize=8)
    ax.set_xlabel("Sampling stride →")
    ax.set_ylabel("Top-1 accuracy (%)" if ds == "autsl" else "")
    ax.set_title(tds_labels[ds], fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")

# Single legend below
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.18), fontsize=8.5,
           framealpha=0.9, edgecolor="0.8")
plt.tight_layout()
savefig(fig, "fig1_aliasing_curves")
plt.close()

# ── FIG 2: Cross-architecture stride sensitivity heatmap ─────────────────
print("Generating Fig 2: Cross-architecture heatmap...")
fig, ax = plt.subplots(figsize=(9, 4.5))

# Matrix: models (rows) × datasets (cols), value = accuracy drop s1→s16
matrix = np.full((len(MODELS), len(DATASETS)), np.nan)
for i, model in enumerate(MODELS):
    for j, ds in enumerate(DATASETS):
        curve = load_stride_curve(model, ds, coverage=100)
        if curve is not None and 1 in curve.index and 16 in curve.index:
            s1 = curve[1] * 100
            s16 = curve[16] * 100
            if s1 > 5:  # exclude collapse
                matrix[i, j] = s1 - s16
            else:
                matrix[i, j] = np.nan  # mark collapse

im = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=80)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Accuracy drop s=1→16 (pp)", fontsize=9)

ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([LABELS[m] for m in MODELS], fontsize=9)

# Annotate cells
for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        v = matrix[i, j]
        if not np.isnan(v):
            color = "white" if v > 45 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
        else:
            ax.text(j, i, "†", ha="center", va="center", fontsize=9, color="gray")

    ax.set_title("Temporal Aliasing Sensitivity — Accuracy Drop (stride 1→16, coverage=100%)\n"
             "† = feature collapse (s1<5%).",
             fontsize=10, pad=10)

# Add family brackets
family_spans = [(0, 3, "CNNs"), (4, 4, "Transformer\n(div-attn)"), (5, 5, "Transformer\n(fact-attn)"),
                (6, 6, "MAE"), (7, 7, "SSM")]
for start, end, name in family_spans:
    ax.annotate("", xy=(-0.55, end+0.4), xytext=(-0.55, start-0.4),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=1.5))

plt.tight_layout()
savefig(fig, "fig2_aliasing_heatmap")
plt.close()

# ── FIG 3: TDS ranking + E3 spectral correlation ─────────────────────────
print("Generating Fig 3: TDS ranking and spectral correlation...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

# Left: TDS ranking bar chart (avg aliasing loss across models, n=8 all datasets)
# VideoMamba included for all datasets; feature-collapsed models (s1<5%) excluded
tds_data = {}
for ds in DATASETS:
    losses = []
    for model in MODELS:
        curve = load_stride_curve(model, ds, coverage=100)
        if curve is not None and 1 in curve.index and 16 in curve.index:
            s1, s16 = curve[1], curve[16]
            if s1 > 0.05:
                losses.append((s1 - s16) * 100)
    if losses:
        tds_data[ds] = np.mean(losses)

tds_sorted = sorted(tds_data.items(), key=lambda x: -x[1])
colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, len(tds_sorted)))
ax1.barh([DATASET_LABELS[d] for d, _ in tds_sorted],
         [v for _, v in tds_sorted], color=colors_bar, edgecolor="white")
ax1.set_xlabel("Mean aliasing loss (pp), stride 1→16")
ax1.set_title("TDS Ranking\n(avg across 8 architectures)", fontsize=10)
ax1.grid(True, axis="x", alpha=0.3, ls="--")
for i, (ds, v) in enumerate(tds_sorted):
    ax1.text(v + 0.5, i, f"{v:.1f}pp", va="center", fontsize=8.5)

# Right: E3 spectral correlation (includes FineGym once flow extraction completes)
try:
    corr = pd.read_csv("evaluations/accv2026/e3_spectral/flow_aliasing_correlation.csv")
    ax2.scatter(corr["pearson_r_abs"], [DATASET_LABELS.get(d, d) for d in corr["dataset"]],
                c=["#E64B35" if r > 0.2 else "#4DBBD5" for r in corr["pearson_r_abs"]],
                s=80, zorder=5, edgecolors="0.3", lw=0.8)
    ax2.axvline(0, color="0.5", lw=1, ls="--")
    ax2.set_xlabel("Pearson r (optical flow ↔ aliasing loss)")
    ax2.set_title("Spectral Correlation\n(flow magnitude vs aliasing sensitivity)", fontsize=10)
    ax2.set_xlim(-0.55, 0.45)
    ax2.grid(True, axis="x", alpha=0.3, ls="--")
    ax2.annotate("Nyquist prediction:\nhigher flow → more aliasing", xy=(0.3, 0.5),
                 xycoords="axes fraction", fontsize=7.5, color="#E64B35")
except Exception as e:
    ax2.text(0.5, 0.5, f"E3 data not found\n{e}", ha="center", va="center",
             transform=ax2.transAxes)

plt.tight_layout()
savefig(fig, "fig3_tds_spectral")
plt.close()

# ── FIG 4: E7 Routing comparison ─────────────────────────────────────────
print("Generating Fig 4: E7 routing comparison...")
try:
    existing = pd.read_csv("evaluations/accv2026/paper_results/paper_table_main_comparison.csv")
    e7 = pd.read_csv("evaluations/accv2026/e7_routing/e7_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    show_model_ds = [("timesformer","ssv2"), ("videomae","ssv2"), ("r2plus1d_18","ssv2")]
    titles = ["TimeSformer / SSv2", "VideoMAE / SSv2", "R2Plus1D / SSv2"]

    for ax, (model, ds), title in zip(axes, show_model_ds, titles):
        ds_label = "SSV2" if ds == "ssv2" else ds.upper()
        model_label = {
            "timesformer": "TimeSformer", "videomae": "VideoMAE",
            "r2plus1d_18": "R2+1D-18"
        }.get(model, model)

        # Fixed baselines
        fixed = existing[
            (existing["model"] == model_label) &
            (existing["dataset"] == ds_label) &
            (existing["method_type"] == "fixed")
        ]
        if not fixed.empty:
            ax.scatter(fixed["avg_frames"], fixed["accuracy"]*100,
                       c="gray", s=60, zorder=5, marker="D", label="Fixed budget")

        # FrameExit
        fe = existing[
            (existing["model"] == model_label) &
            (existing["dataset"] == ds_label) &
            (existing["method_type"] == "frameexit")
        ]
        if not fe.empty:
            ax.plot(fe["avg_frames"], fe["accuracy"]*100, "k--",
                    lw=1.5, marker="s", ms=5, label="FrameExit", alpha=0.7)

        # E7 routing curve
        routing_csv = Path(f"evaluations/accv2026/e7_routing/{model}_{ds}_routing.csv")
        if routing_csv.exists():
            r_df = pd.read_csv(routing_csv)
            ax.plot(r_df["avg_frames"], r_df["accuracy"]*100,
                    color="#E64B35", lw=2, label="E7-Entropy (ours)", zorder=6)

        # Oracle
        oracle = existing[
            (existing["model"] == model_label) &
            (existing["dataset"] == ds_label) &
            (existing["method_type"] == "oracle_knapsack")
        ]
        if not oracle.empty:
            ax.plot(oracle["avg_frames"], oracle["accuracy"]*100, "g--",
                    lw=1.5, marker="^", ms=5, label="Oracle", alpha=0.8)

        ax.set_xlabel("Avg frames used")
        ax.set_ylabel("Top-1 accuracy (%)" if model == "timesformer" else "")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_xlim(2, 20)

    plt.tight_layout()
    savefig(fig, "fig4_routing_comparison")
except Exception as e:
    print(f"  WARNING: Fig 4 error: {e}")
plt.close()

# ── FIG 5: E6 Spatial resolution curves ──────────────────────────────────
print("Generating Fig 5: Spatial resolution curves...")
RESOLUTIONS = [96, 112, 160, 224]
SPATIAL_BASE = Path("evaluations/accv2026/spatial_resolution_sweep")
NATIVE = {"r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
          "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224}

fig, ax = plt.subplots(figsize=(8, 4.5))
for model in MODELS:
    d = SPATIAL_BASE / f"{model}_ssv2"
    points = []
    for res in RESOLUTIONS:
        f = d / f"res{res}_summary.csv"
        if f.exists():
            df_r = pd.read_csv(f)
            if not df_r.empty:
                acc = float(df_r.iloc[0]["top1"]) * 100
                points.append((res, acc))
    if len(points) >= 3:
        xs, ys = zip(*points)
        native = NATIVE[model]
        ls = "--" if native == 112 else "-"
        ax.plot(xs, ys, marker=MARKERS[model], color=COLORS[model],
                label=LABELS[model], lw=1.8, ms=6, ls=ls, alpha=0.9)
        # Mark native resolution
        native_acc = dict(points).get(native)
        if native_acc:
            ax.scatter([native], [native_acc], color=COLORS[model], s=80,
                       zorder=10, edgecolors="black", lw=1.2)

ax.set_xlabel("Spatial resolution (px)")
ax.set_ylabel("Top-1 accuracy (%) on SSv2")
ax.set_title("Spatial Aliasing — Accuracy vs Resolution (SSv2)\n"
             "Filled markers = native resolution; dashed = CNN (native 112px)",
             fontsize=10)
ax.set_xticks(RESOLUTIONS)
ax.legend(fontsize=8, ncol=2, loc="lower right")
ax.grid(True, alpha=0.3, ls="--")
plt.tight_layout()
savefig(fig, "fig5_spatial_resolution")
plt.close()

print("\n=== MAIN FIGURES COMPLETE ===")

# ── SUPPLEMENTARY ─────────────────────────────────────────────────────────
print("\nGenerating Supplementary figures...")

# Sup1: Full E1 heatmaps — one panel per model, all datasets
print("  Sup1: Full E1 heatmaps...")
COVERAGES = [10, 25, 50, 75, 100]
for model in MODELS:
    n_ds = sum(1 for ds in DATASETS
               if (BASE / f"{model}_{ds}" / "sweep_summary.csv").exists())
    if n_ds == 0:
        continue
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(20, 3.5))
    for ax, ds in zip(axes.flat, DATASETS):
        csv = BASE / f"{model}_{ds}" / "sweep_summary.csv"
        if not csv.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv)
        pivot = df.pivot(index="coverage", columns="stride", values="top1")
        pivot = pivot.reindex(index=COVERAGES, columns=STRIDES) * 100
        is_collapse = pivot.loc[100, 1] < 5 if (100 in pivot.index and 1 in pivot.columns) else False
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn" if not is_collapse else "Greys",
                       vmin=0, vmax=pivot.values[~np.isnan(pivot.values)].max() if not is_collapse else 5)
        ax.set_xticks(range(len(STRIDES)))
        ax.set_xticklabels([f"s{s}" for s in STRIDES], fontsize=7)
        ax.set_yticks(range(len(COVERAGES)))
        ax.set_yticklabels([f"{c}%" for c in COVERAGES], fontsize=7)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=8)
        if is_collapse:
            ax.text(2, 2, "NOT\nCOMPARABLE", ha="center", va="center",
                    fontsize=8, color="red", fontweight="bold")
        else:
            for ci, cov in enumerate(COVERAGES):
                for si, stride in enumerate(STRIDES):
                    v = pivot.loc[cov, stride] if (cov in pivot.index and stride in pivot.columns) else np.nan
                    if not np.isnan(v):
                        ax.text(si, ci, f"{v:.0f}", ha="center", va="center",
                                fontsize=5.5, color="white" if v < 20 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{LABELS[model]} — Coverage×Stride accuracy (%) per dataset", fontsize=10)
    plt.tight_layout()
    savefig(fig, f"sup1_heatmap_{model}", folder=OUT_SUPP)
    plt.close()

# Sup2: E2 Levene variance plots
print("  Sup2: E2 variance (Levene)...")
try:
    levene = pd.read_csv("evaluations/accv2026/e2_variance/levene_results.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    sig = levene[levene["significant"] == True]
    insig = levene[levene["significant"] == False]
    ax.scatter(insig["std_s1"], insig["std_s16"], c="gray", alpha=0.5, s=30, label="Not significant")
    scatter = ax.scatter(sig["std_s1"], sig["std_s16"],
                         c=sig["var_ratio_16_over_1"], cmap="Reds", alpha=0.8, s=60,
                         vmin=1, vmax=2.5, label="Significant (p<0.05)", edgecolors="0.3", lw=0.5)
    plt.colorbar(scatter, ax=ax, label="Variance ratio (σ_s16 / σ_s1)")
    ax.plot([0, 0.35], [0, 0.35], "k--", lw=1, alpha=0.5, label="No change line")
    ax.set_xlabel("Inter-class std at stride=1 (dense)")
    ax.set_ylabel("Inter-class std at stride=16 (sparse)")
    ax.set_title("Stride increases inter-class accuracy variance (Levene test)\n"
                 "Points above diagonal = higher variance at stride=16", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    savefig(fig, "sup2_levene_variance", folder=OUT_SUPP)
except Exception as e:
    print(f"    WARNING: {e}")
plt.close()

# Sup3: E4 ANOVA η² bar chart
print("  Sup3: E4 ANOVA...")
try:
    anova = pd.read_csv("evaluations/accv2026/e4_anova/anova_results.csv")
    agg = anova.groupby("model").agg(
        eta2_stride_mean=("eta2_stride","mean"),
        eta2_stride_std=("eta2_stride","std"),
        eta2_cov_mean=("eta2_coverage","mean"),
    ).reset_index().sort_values("eta2_stride_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(agg))
    ax.barh(y, agg["eta2_cov_mean"], color="#4DBBD5", alpha=0.6, label="η²(coverage)")
    ax.barh(y, agg["eta2_stride_mean"], left=agg["eta2_cov_mean"],
            color="#E64B35", alpha=0.8, label="η²(stride)")
    ax.errorbar(agg["eta2_cov_mean"] + agg["eta2_stride_mean"], y,
                xerr=agg["eta2_stride_std"], fmt="none", color="black", capsize=3, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in agg["model"]], fontsize=9)
    ax.set_xlabel("Effect size (η²)")
    ax.set_title("ANOVA Effect Sizes\nη²(stride) + η²(coverage) per model (avg ± std across datasets)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.axvline(0.14, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(0.145, 0.2, "medium", fontsize=7, color="gray", rotation=90)
    ax.grid(True, axis="x", alpha=0.3, ls="--")
    plt.tight_layout()
    savefig(fig, "sup3_anova_eta2", folder=OUT_SUPP)
except Exception as e:
    print(f"    WARNING: {e}")
plt.close()

# Sup4: E5 Taxonomy bar chart
print("  Sup4: E5 taxonomy...")
try:
    tax = pd.read_csv("evaluations/accv2026/e5_taxonomy/taxonomy_summary.csv")
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(20, 4))
    tier_colors = {"High": "#E64B35", "Moderate": "#F39B7F", "Low": "#4DBBD5"}
    for ax, ds in zip(axes.flat, DATASETS):
        sub = tax[tax["dataset"] == ds].sort_values("tier")
        if sub.empty:
            ax.set_visible(False)
            continue
        bars = ax.bar(sub["tier"], sub["mean_abs_drop_pp"],
                      color=[tier_colors.get(t, "gray") for t in sub["tier"]],
                      edgecolor="white", width=0.6)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=9)
        ax.set_ylabel("Aliasing loss (pp)" if ds == "ucf101" else "")
        ax.set_xlabel("")
        for bar, (_, r) in zip(bars, sub.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"n={r['n_classes']}", ha="center", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3, ls="--")
    fig.suptitle("Action Sensitivity Taxonomy: Aliasing loss by tier per dataset",
                 fontsize=10)
    plt.tight_layout()
    savefig(fig, "sup4_taxonomy", folder=OUT_SUPP)
except Exception as e:
    print(f"    WARNING: {e}")
plt.close()

# Sup5: E7 routing curves for all models
print("  Sup5: E7 routing all models...")
try:
    n_models = len(MODELS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, model in zip(axes.flat, MODELS):
        routing_csv = Path(f"evaluations/accv2026/e7_routing/{model}_ssv2_routing.csv")
        if not routing_csv.exists():
            ax.set_visible(False)
            continue
        r_df = pd.read_csv(routing_csv)
        ax.plot(r_df["avg_frames"], r_df["accuracy"]*100,
                color=COLORS[model], lw=2, label="E7-Entropy")
        # Fixed baselines
        ax.axhline(r_df["fixed_cheap_acc"].iloc[0]*100, color="gray", ls=":",
                   lw=1.2, alpha=0.7, label="Fixed 4f")
        ax.axhline(r_df["fixed_dense_acc"].iloc[0]*100, color="gray", ls="--",
                   lw=1.2, alpha=0.7, label="Fixed 16f")
        ax.axhline(r_df["oracle_accuracy"].iloc[0]*100, color="#4DBBD5", ls="-.",
                   lw=1.2, alpha=0.7, label="Oracle")
        ax.set_title(LABELS[model], fontsize=8.5, color=COLORS[model])
        ax.set_xlabel("Avg frames" if model in MODELS[-4:] else "")
        ax.set_ylabel("Accuracy (%)" if model in [MODELS[0], MODELS[4]] else "")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_xlim(3, 17)
    fig.suptitle("Entropy Routing — All Models on SSv2 (4f cheap → 16f dense)",
                 fontsize=11)
    plt.tight_layout()
    savefig(fig, "sup5_routing_all_models", folder=OUT_SUPP)
except Exception as e:
    print(f"    WARNING: {e}")
plt.close()

# Sup6: E10 clip duration
print("  E10 clip duration...")
try:
    dur = pd.read_csv("evaluations/accv2026/e10_duration/duration_summary.csv")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, model in zip(axes.flat, MODELS):
        sub = dur[dur["model"] == model]
        if sub.empty:
            ax.set_visible(False)
            continue
        datasets_avail = sub["dataset"].unique()
        for ds in datasets_avail:
            d_sub = sub[sub["dataset"] == ds].sort_values("duration_bin")
            if len(d_sub) < 2:
                continue
            bin_order = ["<1s","1-3s","3-6s",">6s"]
            d_sub["bin_ord"] = d_sub["duration_bin"].map({b:i for i,b in enumerate(bin_order)})
            d_sub = d_sub.sort_values("bin_ord")
            ax.plot(range(len(d_sub)), d_sub["aliasing_loss_pp"],
                    marker="o", ms=4, label=DATASET_LABELS.get(ds, ds), alpha=0.7)
        ax.set_title(LABELS[model], fontsize=8.5, color=COLORS[model])
        ax.set_xlabel("Clip duration" if model in MODELS[-4:] else "")
        ax.set_ylabel("Aliasing loss (pp)" if model in [MODELS[0],MODELS[4]] else "")
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(["<1s","1-3s","3-6s",">6s"], fontsize=8)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3, ls="--")
        ax.axhline(0, color="0.5", lw=0.8, ls="--")
    fig.suptitle("Clip Duration vs Aliasing Loss\n"
                 "Negative trend: shorter clips alias more (less temporal redundancy)",
                 fontsize=10)
    plt.tight_layout()
    savefig(fig, "sup6_clip_duration", folder=OUT_SUPP)
except Exception as e:
    print(f"    WARNING: {e}")
plt.close()

print("\n" + "="*60)
print("ALL FIGURES GENERATED")
print(f"Main paper: {OUT_MAIN}")
print(f"Supplementary: {OUT_SUPP}")
print()
print("Main figures:")
for f in sorted(OUT_MAIN.glob("*.pdf")):
    print(f"  {f.name}")
print()
print("Supplementary figures:")
for f in sorted(OUT_SUPP.glob("*.pdf")):
    print(f"  {f.name}")
