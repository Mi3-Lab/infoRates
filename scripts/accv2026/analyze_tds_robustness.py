#!/usr/bin/env python3
"""TDS robustness analysis for reviewer response.

Answers three reviewer questions using EXISTING sweep data (no retraining):
  1. How does the TDS ranking change if CNNs are removed?
  2. How does the TDS ranking change if only Transformers (or Transformer+SSM) remain?
  3. What are bootstrap confidence intervals for the TDS ranking, and is the
     FineGym vs. AUTSL "co-equal #1/#2" claim statistically supported?

Mirrors the exact TDS definition in dashboard/app.py::compute_tds() and
paper/main.tex Eq. 1: architecture-averaged accuracy drop from stride=1 to
stride=16 at coverage=100%, excluding feature-collapsed models (acc@s=1 < 5%).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dashboard/data"
SWEEP_ROOT = ROOT / "evaluations/accv2026/coverage_stride_sweep"

MODEL_KEYS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
              "timesformer", "vivit", "videomae", "videomamba"]
FAMILIES = {
    "r3d_18": "CNN", "mc3_18": "CNN", "r2plus1d_18": "CNN", "slowfast_r50": "Dual-CNN",
    "timesformer": "Transformer", "vivit": "Transformer", "videomae": "Transformer",
    "videomamba": "SSM",
}
DS_KEYS = ["autsl", "diving48", "ssv2", "hmdb51", "driveact",
           "epic_kitchens", "ucf101", "finegym"]
NATIVE = {"r3d_18": 112, "mc3_18": 112, "r2plus1d_18": 112, "slowfast_r50": 224,
          "timesformer": 224, "vivit": 224, "videomae": 224, "videomamba": 224}


def load_sweeps() -> pd.DataFrame:
    """Same loader as dashboard/app.py::load_sweeps() — primary CSV + trainres fallback."""
    f = DATA / "sweep_summary.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    present = set() if df.empty else set(zip(df["model"], df["dataset"]))
    extra_rows = []
    if SWEEP_ROOT.exists():
        for mk, native in NATIVE.items():
            for ds in DS_KEYS:
                if (mk, ds) in present:
                    continue
                trainres_csv = SWEEP_ROOT / f"{mk}_{ds}_trainres{native}" / "sweep_summary.csv"
                if trainres_csv.exists():
                    tmp = pd.read_csv(trainres_csv)
                    tmp["model"] = mk
                    tmp["dataset"] = ds
                    extra_rows.append(tmp)
    if extra_rows:
        df = pd.concat([df] + extra_rows, ignore_index=True) if not df.empty \
            else pd.concat(extra_rows, ignore_index=True)
    df["acc"] = df["top1"] * 100
    return df


def compute_tds(df_sweep: pd.DataFrame, model_pool: list[str]) -> dict[str, float]:
    """Exact same logic as dashboard/app.py::compute_tds(), restricted to model_pool."""
    tds = {}
    for ds in DS_KEYS:
        drops = []
        sub = df_sweep[(df_sweep.dataset == ds) & (df_sweep.coverage == 100)]
        for m in model_pool:
            s1 = sub[(sub.model == m) & (sub.stride == 1)]["acc"]
            s16 = sub[(sub.model == m) & (sub.stride == 16)]["acc"]
            if s1.empty or s16.empty:
                continue
            if s1.values[0] < 5:
                continue  # feature collapse
            drops.append(s1.values[0] - s16.values[0])
        tds[ds] = round(float(np.mean(drops)), 2) if drops else 0.0
    return tds


def spearman(a: dict, b: dict) -> float:
    keys = list(a.keys())
    ra = pd.Series([a[k] for k in keys]).rank()
    rb = pd.Series([b[k] for k in keys]).rank()
    return float(ra.corr(rb, method="pearson"))  # Pearson on ranks == Spearman


def per_model_drops(df_sweep: pd.DataFrame) -> dict[str, dict[str, float]]:
    """drops[model][dataset] = acc(s=1) - acc(s=16) at cov=100, or None if collapsed."""
    out = {m: {} for m in MODEL_KEYS}
    for ds in DS_KEYS:
        sub = df_sweep[(df_sweep.dataset == ds) & (df_sweep.coverage == 100)]
        for m in MODEL_KEYS:
            s1 = sub[(sub.model == m) & (sub.stride == 1)]["acc"]
            s16 = sub[(sub.model == m) & (sub.stride == 16)]["acc"]
            if s1.empty or s16.empty or s1.values[0] < 5:
                continue
            out[m][ds] = float(s1.values[0] - s16.values[0])
    return out


def bootstrap_tds(drops: dict[str, dict[str, float]], n_boot: int = 10000,
                   seed: int = 0) -> pd.DataFrame:
    """Resample architectures with replacement; recompute TDS per dataset each time."""
    rng = np.random.default_rng(seed)
    models = list(drops.keys())
    boot_vals = {ds: [] for ds in DS_KEYS}
    for _ in range(n_boot):
        sample = rng.choice(models, size=len(models), replace=True)
        for ds in DS_KEYS:
            vals = [drops[m][ds] for m in sample if ds in drops[m]]
            if vals:
                boot_vals[ds].append(np.mean(vals))
    rows = []
    for ds in DS_KEYS:
        arr = np.array(boot_vals[ds])
        rows.append({
            "dataset": ds,
            "tds_mean": round(float(arr.mean()), 2),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 2),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 2),
        })
    return pd.DataFrame(rows).sort_values("tds_mean", ascending=False).reset_index(drop=True)


def bootstrap_gap(drops: dict[str, dict[str, float]], ds_a: str, ds_b: str,
                   n_boot: int = 10000, seed: int = 1) -> dict:
    """Bootstrap CI for TDS(ds_a) - TDS(ds_b), e.g. FineGym vs AUTSL."""
    rng = np.random.default_rng(seed)
    models = list(drops.keys())
    gaps = []
    for _ in range(n_boot):
        sample = rng.choice(models, size=len(models), replace=True)
        va = [drops[m][ds_a] for m in sample if ds_a in drops[m]]
        vb = [drops[m][ds_b] for m in sample if ds_b in drops[m]]
        if va and vb:
            gaps.append(np.mean(va) - np.mean(vb))
    gaps = np.array(gaps)
    return {
        "gap_mean": round(float(gaps.mean()), 2),
        "ci_lo": round(float(np.percentile(gaps, 2.5)), 2),
        "ci_hi": round(float(np.percentile(gaps, 97.5)), 2),
        "pct_favoring_a": round(float((gaps > 0).mean()) * 100, 1),
    }


def main():
    df = load_sweeps()
    print(f"Loaded sweep data: {len(df)} rows, models={sorted(df.model.unique())}, "
          f"datasets={sorted(df.dataset.unique())}\n")

    pools = {
        "Full (n=8)": MODEL_KEYS,
        "CNN-only (n=4)": [m for m in MODEL_KEYS if FAMILIES[m] in ("CNN", "Dual-CNN")],
        "Transformer-only (n=3)": [m for m in MODEL_KEYS if FAMILIES[m] == "Transformer"],
        "Transformer+SSM (n=4)": [m for m in MODEL_KEYS if FAMILIES[m] in ("Transformer", "SSM")],
    }

    tds_by_pool = {name: compute_tds(df, pool) for name, pool in pools.items()}

    print("=" * 80)
    print("TDS per dataset, by architecture pool")
    print("=" * 80)
    table = pd.DataFrame(tds_by_pool)
    table = table.loc[table["Full (n=8)"].sort_values(ascending=False).index]
    print(table.to_string())
    print()

    print("=" * 80)
    print("Ranking stability: Spearman correlation vs. Full (n=8) pool")
    print("=" * 80)
    full = tds_by_pool["Full (n=8)"]
    for name, tds in tds_by_pool.items():
        if name == "Full (n=8)":
            continue
        rho = spearman(full, tds)
        print(f"  {name:28s} rho={rho:.3f}  pool={pools[name]}")
    print()

    drops = per_model_drops(df)
    print("=" * 80)
    print("Bootstrap 95% CI for TDS ranking (resample 8 architectures w/ replacement, B=10000)")
    print("=" * 80)
    boot_table = bootstrap_tds(drops, n_boot=10000)
    print(boot_table.to_string(index=False))
    print()

    print("=" * 80)
    print("Bootstrap CI for FineGym vs AUTSL gap (co-equal #1/#2 claim)")
    print("=" * 80)
    gap = bootstrap_gap(drops, "finegym", "autsl", n_boot=10000)
    print(f"  TDS(FineGym) - TDS(AUTSL): mean={gap['gap_mean']:+.2f}pp  "
          f"95% CI=[{gap['ci_lo']:+.2f}, {gap['ci_hi']:+.2f}]  "
          f"P(FineGym > AUTSL)={gap['pct_favoring_a']:.1f}%")
    print("  -> CI includes 0:", gap["ci_lo"] <= 0 <= gap["ci_hi"],
          "(supports 'co-equal' framing if True)")
    print()

    out = {
        "tds_by_pool": tds_by_pool,
        "spearman_vs_full": {name: spearman(full, tds) for name, tds in tds_by_pool.items()
                              if name != "Full (n=8)"},
        "bootstrap_ci": boot_table.to_dict(orient="records"),
        "finegym_vs_autsl_gap": gap,
    }
    out_path = ROOT / "evaluations/accv2026/tds_robustness_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    boot_table.to_csv(ROOT / "evaluations/accv2026/tds_bootstrap_ci.csv", index=False)
    print(f"Saved: {out_path}")
    print(f"Saved: {ROOT / 'evaluations/accv2026/tds_bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
