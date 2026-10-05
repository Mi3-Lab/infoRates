"""Figures for the IEEE FG 2027 version of the paper.

Same data and loaders as scripts/accv2026/generate_paper_figures.py, with the
differences the two-column IEEE layout needs:

  * no figure numbers or titles baked into the image (the caption carries them);
  * the stride-drop heatmap covers all 8 datasets and is sized for one column;
  * the cascade is labelled as a confidence cascade (it thresholds max-prob);
  * supplementary routing and clip-duration figures are regenerated to match.

Usage:  .venv/bin/python scripts/fg2027/make_figures.py
Output: paper/fg2027/images/
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/fg2027/images"
OUT.mkdir(parents=True, exist_ok=True)
EVAL = ROOT / "evaluations/accv2026"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

COLORS = {
    "r3d_18": "#E64B35", "mc3_18": "#F39B7F", "r2plus1d_18": "#FF7F50",
    "slowfast_r50": "#C0392B", "timesformer": "#4DBBD5", "vivit": "#8FBC8F",
    "videomae": "#1F78B4", "videomamba": "#7B2D8B",
}
LABELS = {
    "r3d_18": "R3D-18", "mc3_18": "MC3-18", "r2plus1d_18": "R(2+1)D",
    "slowfast_r50": "SlowFast", "timesformer": "TimeSformer", "vivit": "ViViT",
    "videomae": "VideoMAE", "videomamba": "VideoMamba",
}
MARKERS = {
    "r3d_18": "o", "mc3_18": "s", "r2plus1d_18": "^", "slowfast_r50": "D",
    "timesformer": "o", "vivit": "s", "videomae": "^", "videomamba": "*",
}
MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact",
            "epic_kitchens", "finegym"]
DATASET_LABELS = {
    "ucf101": "UCF-101", "ssv2": "SSv2", "hmdb51": "HMDB-51",
    "diving48": "Diving-48", "autsl": "AUTSL", "driveact": "DriveAct",
    "epic_kitchens": "EPIC-Kitchens", "finegym": "FineGym",
}
STRIDES = [1, 2, 4, 8, 16]
NATIVE = {"r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
          "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224}
SWEEP = EVAL / "coverage_stride_sweep"
_DASH = pd.read_csv(ROOT / "dashboard/data/sweep_summary.csv")


def load_stride_curve(model, dataset, coverage=100):
    """Accuracy vs stride at one coverage; dashboard CSV first, then sweep dirs."""
    sub = _DASH[(_DASH.model == model) & (_DASH.dataset == dataset)
                & (_DASH.coverage == coverage)].sort_values("stride")
    if not sub.empty and {1, 16} <= set(sub.stride):
        return sub.set_index("stride")["top1"]
    for suffix in ("", f"_trainres{NATIVE[model]}"):
        csv = SWEEP / f"{model}_{dataset}{suffix}" / "sweep_summary.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            sub = df[df.coverage == coverage].sort_values("stride")
            if not sub.empty:
                return sub.set_index("stride")["top1"]
    return None


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  {OUT.relative_to(ROOT)}/{name}.pdf")


# Stride-drop matrix, shared by the heatmap and the TDS ranking.
drop = pd.DataFrame(np.nan, index=MODELS, columns=DATASETS)
for m in MODELS:
    for d in DATASETS:
        c = load_stride_curve(m, d)
        if c is not None and {1, 16} <= set(c.index) and c[1] > 0.05:
            drop.loc[m, d] = (c[1] - c[16]) * 100
tds = drop.clip(lower=0).mean().sort_values(ascending=False)
model_order = drop.mean(axis=1).sort_values().index.tolist()
print("TDS:", tds.round(1).to_dict())
print("mean drop per model:", drop.mean(axis=1).round(1).to_dict())

# ── Stride curves, three datasets spanning the TDS range (full width) ─────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
for ax, ds in zip(axes, ["autsl", "ssv2", "ucf101"]):
    for m in MODELS:
        c = load_stride_curve(m, ds)
        if c is None:
            continue
        ax.plot(STRIDES, [c.get(s, np.nan) * 100 for s in STRIDES],
                marker=MARKERS[m], color=COLORS[m], label=LABELS[m],
                lw=1.6, ms=4.5, alpha=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(STRIDES)
    ax.set_xticklabels([str(s) for s in STRIDES])
    ax.set_xlabel("Sampling stride $s$")
    ax.set_title(f"{DATASET_LABELS[ds]} (TDS = {tds[ds]:.1f} pp)")
    ax.grid(True, alpha=0.3, ls="--")
axes[0].set_ylabel("Top-1 accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.07),
           framealpha=0.9, edgecolor="0.8")
fig.tight_layout()
save(fig, "fig_stride_curves")

# ── Stride-drop heatmap, all 8 datasets (one column) ──────────────────────
ds_order = tds.index[::-1].tolist()
mat = drop.loc[model_order, ds_order].values
fig, ax = plt.subplots(figsize=(4.6, 3.1))
im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=80)
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
cb.set_label("Accuracy drop, $s$=1$\\to$16 (pp)", fontsize=8)
ax.set_xticks(range(len(ds_order)))
ax.set_xticklabels([DATASET_LABELS[d] for d in ds_order], rotation=35, ha="right")
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels([LABELS[m] for m in model_order])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 45 else "black")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
save(fig, "fig_drop_heatmap")

# ── TDS ranking + flow correlation (one column) ──────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.9, 2.5), gridspec_kw={"wspace": 0.75})
order = tds.index[::-1]
a1.barh([DATASET_LABELS[d] for d in order], tds[order],
        color=plt.cm.Reds(np.linspace(0.3, 0.9, len(order))), edgecolor="white")
for i, d in enumerate(order):
    a1.text(tds[d] + 1, i, f"{tds[d]:.1f}", va="center", fontsize=7.5)
a1.set_xlim(0, 70)
a1.set_xlabel("TDS (pp)")
a1.set_title("(a) Temporal demand")
a1.grid(True, axis="x", alpha=0.3, ls="--")
corr = pd.read_csv(EVAL / "e3_spectral/flow_aliasing_correlation.csv")
corr = corr.set_index("dataset").loc[list(order)]
a2.scatter(corr.pearson_r_abs, range(len(corr)),
           c=["#E64B35" if s else "0.6" for s in corr.significant],
           s=45, zorder=5, edgecolors="0.3", lw=0.6)
a2.axvline(0, color="0.5", lw=1, ls="--")
a2.set_yticks(range(len(corr)))
a2.set_yticklabels([DATASET_LABELS[d] for d in corr.index])
a2.set_xlim(-0.6, 0.45)
a2.set_xlabel("Pearson $r$ (flow vs. loss)")
a2.set_title("(b) Flow correlation")
a2.grid(True, axis="x", alpha=0.3, ls="--")
save(fig, "fig_tds_spectral")

# ── Spatial resolution on SSv2, no retraining (one column) ───────────────
RES = [96, 112, 160, 224]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for m in MODELS:
    pts = []
    for r in RES:
        f = EVAL / "spatial_resolution_sweep" / f"{m}_ssv2" / f"res{r}_summary.csv"
        if f.exists():
            df = pd.read_csv(f)
            if not df.empty:
                pts.append((r, float(df.iloc[0]["top1"]) * 100))
    if len(pts) < 3:
        continue
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=LABELS[m],
            lw=1.6, ms=5, ls="--" if NATIVE[m] == 112 else "-", alpha=0.9)
    if NATIVE[m] in dict(pts):
        ax.scatter([NATIVE[m]], [dict(pts)[NATIVE[m]]], color=COLORS[m], s=55,
                   zorder=10, edgecolors="black", lw=1.1)
ax.set_xticks(RES)
ax.set_xlabel("Input resolution (px)")
ax.set_ylabel("Top-1 accuracy (%), SSv2")
ax.legend(ncol=2, loc="lower center", fontsize=7)
ax.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
save(fig, "fig_spatial_resolution")

# ── Confidence cascade on SSv2 (one column, three panels) ────────────────
# Reference lines come from the same routing CSV as the curve (and as the
# routing table), so every element of a panel shares one evaluation protocol.
existing = pd.read_csv(EVAL / "paper_results/paper_table_main_comparison.csv")
panel = [("timesformer", "TimeSformer"), ("videomae", "VideoMAE"),
         ("r2plus1d_18", "R2+1D-18")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, (m, mlabel) in zip(axes, panel):
    r = pd.read_csv(EVAL / "e7_routing" / f"{m}_ssv2_routing.csv")
    ax.axhline(r.oracle_accuracy.iloc[0] * 100, color="g", ls="-.", lw=1.1,
               label="Oracle")
    ax.axhline(r.fixed_dense_acc.iloc[0] * 100, color="gray", ls="--", lw=1.1,
               label="Fixed, dense")
    ax.axhline(r.fixed_cheap_acc.iloc[0] * 100, color="gray", ls=":", lw=1.1,
               label="Fixed, cheap")
    ax.plot(r.avg_frames, r.accuracy * 100, color="#E64B35", lw=1.8, zorder=6,
            label="Cascade (ours)")
    fe = existing[(existing.model == mlabel) & (existing.dataset == "SSV2")
                  & (existing.method_type == "frameexit")]
    if not fe.empty:
        ax.plot(fe.avg_frames, fe.accuracy * 100, "k--", lw=1.1, marker="s",
                ms=3.5, alpha=0.7, label="FrameExit")
    ax.set_title(LABELS[m])
    ax.set_xlabel("Avg. frames")
    ax.set_xlim(3, 17)
    ax.grid(True, alpha=0.3, ls="--")
axes[0].set_ylabel("Top-1 accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1),
           framealpha=0.9, edgecolor="0.8")
fig.tight_layout()
save(fig, "fig_cascade")

# ── Supplementary: cascade curves for every model with routing data ──────
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
for ax, m in zip(axes.flat, MODELS):
    rc = EVAL / "e7_routing" / f"{m}_ssv2_routing.csv"
    if not rc.exists():
        ax.set_visible(False)
        continue
    r = pd.read_csv(rc)
    ax.plot(r.avg_frames, r.accuracy * 100, color=COLORS[m], lw=2, label="Cascade")
    ax.axhline(r.fixed_cheap_acc.iloc[0] * 100, color="gray", ls=":", lw=1.2,
               label="Fixed, cheap")
    ax.axhline(r.fixed_dense_acc.iloc[0] * 100, color="gray", ls="--", lw=1.2,
               label="Fixed, dense")
    ax.axhline(r.oracle_accuracy.iloc[0] * 100, color="#4DBBD5", ls="-.", lw=1.2,
               label="Oracle")
    ax.set_title(LABELS[m], color=COLORS[m])
    ax.set_xlabel("Avg. frames")
    ax.set_xlim(3, 17)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, alpha=0.3, ls="--")
for ax in axes[:, 0]:
    ax.set_ylabel("Top-1 accuracy (%)")
fig.tight_layout()
save(fig, "sup_cascade_all_models")

# ── Supplementary: clip duration vs stride-induced loss ──────────────────
dur = pd.read_csv(EVAL / "e10_duration/duration_summary.csv")
bins = ["<1s", "1-3s", "3-6s", ">6s"]
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
for ax, m in zip(axes.flat, MODELS):
    sub = dur[dur.model == m]
    if sub.empty:
        ax.set_visible(False)
        continue
    for ds in sub.dataset.unique():
        d = sub[sub.dataset == ds].copy()
        d["o"] = d.duration_bin.map({b: i for i, b in enumerate(bins)})
        d = d.sort_values("o")
        if len(d) >= 2:
            ax.plot(d.o, d.aliasing_loss_pp, marker="o", ms=4, alpha=0.75,
                    label=DATASET_LABELS.get(ds, ds))
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_title(LABELS[m], color=COLORS[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels(bins)
    ax.set_xlabel("Clip duration")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, ls="--")
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy drop, $s$=1$\\to$16 (pp)")
fig.tight_layout()
save(fig, "sup_clip_duration")

# ── Supplementary: full coverage x stride heatmaps, one figure per model ──
COVERAGES = [10, 25, 50, 75, 100]


def load_grid(model, dataset):
    """5x5 accuracy grid (coverage rows, stride columns), same priority as load_stride_curve."""
    sub = _DASH[(_DASH.model == model) & (_DASH.dataset == dataset)]
    if len(sub) < 25:
        sub = None
        for suffix in ("", f"_trainres{NATIVE[model]}"):
            csv = SWEEP / f"{model}_{dataset}{suffix}" / "sweep_summary.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                if len(df) >= 25:
                    sub = df
                    break
    if sub is None:
        return None
    return sub.pivot_table(index="coverage", columns="stride", values="top1").loc[COVERAGES, STRIDES] * 100


ds_by_tds = tds.index.tolist()
for m in MODELS:
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    for ax, ds in zip(axes.flat, ds_by_tds):
        g = load_grid(m, ds)
        if g is None:
            ax.set_visible(False)
            print(f"  [missing grid] {m}/{ds}")
            continue
        im = ax.imshow(g.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{g.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"s{s}" for s in STRIDES])
        ax.set_yticks(range(5)); ax.set_yticklabels([f"{c}%" for c in COVERAGES])
        ax.set_title(DATASET_LABELS[ds])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015)
    cb.set_label("Top-1 accuracy (%)")
    save(fig, f"sup_heatmap_{m}")

# ── Supplementary: Levene variance inflation ─────────────────────────────
lev = pd.read_csv(EVAL / "e2_variance/levene_results.csv")
print(f"Levene: {100 * lev.significant.mean():.0f}% of {len(lev)} pairs significant")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
sig, insig = lev[lev.significant], lev[~lev.significant]
ax.scatter(insig.std_s1, insig.std_s16, c="gray", alpha=0.5, s=25, label="Not significant")
sc = ax.scatter(sig.std_s1, sig.std_s16, c=sig.var_ratio_16_over_1, cmap="Reds",
                vmin=1, vmax=2.5, s=45, alpha=0.85, edgecolors="0.3", lw=0.5,
                label="Significant ($p<0.05$)")
fig.colorbar(sc, ax=ax, label="Variance ratio (stride 16 / stride 1)")
lim = max(lev.std_s1.max(), lev.std_s16.max()) * 1.03
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="No change")
ax.set_xlabel("Inter-class std at stride 1")
ax.set_ylabel("Inter-class std at stride 16")
ax.legend()
ax.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
save(fig, "sup_levene_variance")

# ── Supplementary: between-cell ANOVA effect sizes ───────────────────────
an = pd.read_csv(EVAL / "e4_anova/anova_results.csv")
agg = an.groupby("model").agg(st=("eta2_stride", "mean"), sd=("eta2_stride", "std"),
                              cv=("eta2_coverage", "mean")).sort_values("st")
print("between-cell eta2:", agg.round(2).to_dict("index"))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
y = range(len(agg))
ax.barh(y, agg.cv, color="#4DBBD5", alpha=0.6, label="$\\eta^2$ (coverage)")
ax.barh(y, agg.st, left=agg.cv, color="#E64B35", alpha=0.8, label="$\\eta^2$ (stride)")
ax.errorbar(agg.cv + agg.st, y, xerr=agg.sd, fmt="none", color="black", capsize=3, lw=1)
ax.set_yticks(y)
ax.set_yticklabels([LABELS[m] for m in agg.index])
ax.set_xlabel("Effect size ($\\eta^2$)")
ax.legend(loc="lower left")
ax.grid(True, axis="x", alpha=0.3, ls="--")
fig.tight_layout()
save(fig, "sup_anova_eta2")

# ── Supplementary: taxonomy tiers, all 8 datasets ────────────────────────
tax = pd.read_csv(EVAL / "e5_taxonomy/taxonomy_summary.csv")
tier_order = ["Low", "Moderate", "High"]
tier_colors = {"High": "#E64B35", "Moderate": "#F39B7F", "Low": "#4DBBD5"}
fig, axes = plt.subplots(2, 4, figsize=(11, 5))
for ax, ds in zip(axes.flat, ds_by_tds):
    sub = tax[tax.dataset == ds].set_index("tier").reindex(tier_order).dropna(how="all")
    if sub.empty:
        ax.set_visible(False)
        print(f"  [missing taxonomy] {ds}")
        continue
    bars = ax.bar(sub.index, sub.mean_abs_drop_pp, width=0.6, edgecolor="white",
                  color=[tier_colors[t] for t in sub.index])
    for b, (_, r) in zip(bars, sub.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.5,
                f"n={int(r.n_classes)}", ha="center", fontsize=7)
    ax.set_title(DATASET_LABELS[ds])
    ax.grid(True, axis="y", alpha=0.3, ls="--")
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy drop, $s$=1$\\to$16 (pp)")
fig.tight_layout()
save(fig, "sup_taxonomy")
