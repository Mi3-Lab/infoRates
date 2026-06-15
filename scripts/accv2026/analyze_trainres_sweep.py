#!/usr/bin/env python3
"""Analyze stride×coverage×resolution sweep results.

Computes:
  - TDS (Temporal Demand Score) per model/dataset/training-resolution
  - Relative TDS: normalized by native-resolution TDS
  - Generates heatmaps and 3D surface plots

Usage:
    python analyze_trainres_sweep.py [--output-dir evaluations/accv2026/trainres_analysis]
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "evaluations/accv2026/coverage_stride_sweep"
RESOLUTIONS = [96, 112, 160, 224]

MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
          "timesformer", "vivit", "videomae", "videomamba"]
DATASETS = ["ucf101", "ssv2", "hmdb51", "diving48", "autsl", "driveact", "epic_kitchens"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sweep(model: str, dataset: str, train_res: int) -> pd.DataFrame | None:
    d = SWEEP_DIR / f"{model}_{dataset}_trainres{train_res}"
    csv = d / "sweep_summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["model"] = model
    df["dataset"] = dataset
    df["train_res"] = train_res
    return df


def tds(df: pd.DataFrame) -> float:
    """Mean top1 drop from stride=1 to stride=16 at coverage=100%."""
    at100 = df[df["coverage"] == 100].copy()
    if at100.empty:
        return np.nan
    s1  = at100[at100["stride"] == 1]["top1"].values
    s16 = at100[at100["stride"] == 16]["top1"].values
    if len(s1) == 0 or len(s16) == 0:
        return np.nan
    return float(s1[0] - s16[0])


def collect_all() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for ds in DATASETS:
            for r in RESOLUTIONS:
                df = load_sweep(m, ds, r)
                if df is None:
                    continue
                t = tds(df)
                peak = float(df[df["stride"] == 1]["top1"].max())
                rows.append(dict(model=m, dataset=ds, train_res=r, tds=t, peak_top1=peak))
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_tds_vs_resolution(agg: pd.DataFrame, out_dir: Path):
    """Line plot: TDS vs training resolution, one line per model, faceted by dataset."""
    datasets = sorted(agg["dataset"].unique())
    models = sorted(agg["model"].unique())
    n = len(datasets)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # type: ignore

    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = agg[agg["dataset"] == ds]
        for j, m in enumerate(models):
            ms = sub[sub["model"] == m].sort_values("train_res")
            if ms.empty:
                continue
            ax.plot(ms["train_res"], ms["tds"] * 100, marker="o",
                    color=colors[j], label=m, linewidth=1.5)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Training resolution (px)")
        ax.set_ylabel("TDS (%)")
        ax.set_xticks(RESOLUTIONS)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    # hide empty subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Temporal Demand Score vs Training Resolution", fontsize=13, y=1.01)
    fig.tight_layout()
    out = out_dir / "tds_vs_resolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_heatmap_model(agg: pd.DataFrame, out_dir: Path):
    """Heatmap: model × dataset, cell = TDS at each train_res, faceted by resolution."""
    for res in RESOLUTIONS:
        sub = agg[agg["train_res"] == res].copy()
        if sub.empty:
            continue
        pivot = sub.pivot(index="model", columns="dataset", values="tds") * 100
        fig, ax = plt.subplots(figsize=(len(DATASETS) * 1.4 + 1, len(MODELS) * 0.8 + 1))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=max(30, float(pivot.max().max())))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color="white" if v > 15 else "black")
        plt.colorbar(im, ax=ax, label="TDS (%)")
        ax.set_title(f"TDS at train_res={res}px", fontsize=11)
        fig.tight_layout()
        out = out_dir / f"heatmap_tds_{res}px.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)


def plot_relative_tds(agg: pd.DataFrame, out_dir: Path):
    """Bar: relative TDS (vs 224px baseline) — does lower resolution reduce TDS?"""
    baseline = agg[agg["train_res"] == 224][["model", "dataset", "tds"]].rename(
        columns={"tds": "tds_224"})
    merged = agg.merge(baseline, on=["model", "dataset"], how="left")
    merged["rel_tds"] = (merged["tds"] - merged["tds_224"]) / merged["tds_224"].abs().clip(1e-4)

    models = sorted(agg["model"].unique())
    ress = [r for r in RESOLUTIONS if r != 224]

    fig, axes = plt.subplots(1, len(ress), figsize=(5 * len(ress), 4), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # type: ignore

    for i, r in enumerate(ress):
        ax = axes[i]
        sub = merged[merged["train_res"] == r]
        for j, m in enumerate(models):
            ms = sub[sub["model"] == m]["rel_tds"].values
            if len(ms) == 0:
                continue
            ax.bar(j + np.random.uniform(-0.1, 0.1), ms.mean(), width=0.6,
                   color=colors[j], alpha=0.8, label=m if i == 0 else "")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"train_res={r}px vs 224px")
        ax.set_xlabel("Model")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split("_")[0] for m in models], rotation=45, ha="right")
        ax.set_ylabel("Relative ΔTDS" if i == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.12, 1))
    fig.suptitle("Relative TDS change vs 224px baseline (negative = more temporally robust)",
                 fontsize=11)
    fig.tight_layout()
    out = out_dir / "relative_tds_vs_224.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def print_summary(agg: pd.DataFrame):
    print("\n=== TDS summary (%) by model × train_res ===")
    pivot = agg.groupby(["model", "train_res"])["tds"].mean().unstack("train_res") * 100
    print(pivot.round(1).to_string())

    print("\n=== Peak top1 (%) by model × train_res ===")
    pivot2 = agg.groupby(["model", "train_res"])["peak_top1"].mean().unstack("train_res") * 100
    print(pivot2.round(1).to_string())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
              ROOT / "evaluations/accv2026/trainres_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting results...")
    agg = collect_all()
    if agg.empty:
        print("[ERROR] No sweep results found. Run sweep first.")
        return

    n_total = len(MODELS) * len(DATASETS) * len(RESOLUTIONS)
    print(f"Found {len(agg)}/{n_total} model×dataset×resolution combinations")
    incomplete = n_total - len(agg)
    if incomplete:
        print(f"  ({incomplete} missing — sweep still running)")

    agg.to_csv(out_dir / "trainres_tds_summary.csv", index=False)
    print(f"Saved: {out_dir / 'trainres_tds_summary.csv'}")

    print_summary(agg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(agg) >= 4:
            plot_tds_vs_resolution(agg, out_dir)
        if len(agg) >= 8:
            plot_heatmap_model(agg, out_dir)
        if (agg["train_res"] == 224).any():
            plot_relative_tds(agg, out_dir)


if __name__ == "__main__":
    main()
