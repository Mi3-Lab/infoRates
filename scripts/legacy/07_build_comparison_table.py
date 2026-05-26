#!/usr/bin/env python3
"""Build the main ACCV 2026 comparison table: fixed-budget accuracy curves + temporal AUC.

Usage:
    python scripts/accv2026/06_build_comparison_table.py [--out evaluations/accv2026/metrics/comparison_table.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ARCHITECTURE_MAP = {
    "timesformer": "Transformer",
    "videomae": "Transformer",
    "vivit": "Transformer",
    "r3d18": "3D CNN",
    "r2plus1d": "3D CNN",
    "mc3": "3D CNN",
    "slowfast": "SlowFast",
}

MODEL_DISPLAY = {
    "timesformer_ssv2_10k_e1": "TimeSformer (10k/1ep)",
    "videomae_ssv2_10k_e1_a100ddp": "VideoMAE (10k/1ep)",
    "vivit_ssv2_5k_e1_a100ddp": "ViViT (5k/1ep)",
    "r3d18_ssv2_5k_e1_a100ddp": "R3D-18 (5k/1ep)",
    "mc3_18_ssv2_5k_e1_a100": "MC3-18 (5k/1ep)",
    "r2plus1d_18_ssv2_5k_e1_a100": "R(2+1)D-18 (5k/1ep)",
    "slowfast_r50_ssv2_5k_e1_a100": "SlowFast R50 (5k/1ep)",
}


def detect_arch(model_dir: str) -> str:
    s = model_dir.lower()
    for key, arch in ARCHITECTURE_MAP.items():
        if key in s:
            return arch
    return "Unknown"


def build_table(base_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("smoke"):
            continue

        summary_files = sorted(model_dir.glob("*_fixed_budget_summary.csv"))
        metrics_file = model_dir / "temporal_metrics.csv"
        if not summary_files:
            continue

        df = pd.read_csv(summary_files[0])
        if "budget" not in df.columns or "top1" not in df.columns:
            continue

        budget_map = {int(r["budget"]): float(r["top1"]) for _, r in df.iterrows()}

        auc = None
        critical_budget = None
        if metrics_file.exists():
            m = pd.read_csv(metrics_file)
            auc_col = next((c for c in m.columns if "auc" in c.lower()), None)
            budget_col = next((c for c in m.columns if "critical" in c.lower() and "budget" in c.lower()), None)
            if auc_col:
                auc = float(m[auc_col].iloc[0])
            if budget_col:
                critical_budget = m[budget_col].iloc[0]

        row = {
            "model_dir": model_dir.name,
            "model": MODEL_DISPLAY.get(model_dir.name, model_dir.name),
            "architecture": detect_arch(model_dir.name),
            "top1_4f": budget_map.get(4),
            "top1_8f": budget_map.get(8),
            "top1_16f": budget_map.get(16),
            "top1_32f": budget_map.get(32),
            "temporal_auc": auc,
            "critical_budget": critical_budget,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    cols = ["model", "architecture", "top1_4f", "top1_8f", "top1_16f", "top1_32f", "temporal_auc", "critical_budget"]
    display = df[[c for c in cols if c in df.columns]].copy()

    for c in ["top1_4f", "top1_8f", "top1_16f", "top1_32f", "temporal_auc"]:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")

    print("\n=== ACCV 2026 — Pilot Comparison Table (SSV2 val_5_per_class) ===\n")
    print(display.to_string(index=False))
    print()

    print("--- Ranking by Temporal AUC ---")
    ranked = df.dropna(subset=["temporal_auc"]).sort_values("temporal_auc", ascending=False)
    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {r['model']:40s}  AUC={r['temporal_auc']:.4f}  critical_budget={r['critical_budget']}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="evaluations/accv2026/fixed_budget")
    parser.add_argument("--out", default="evaluations/accv2026/metrics/comparison_table.csv")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_table(base_dir)
    if df.empty:
        print("No completed evaluations found.")
        return

    print_table(df)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
