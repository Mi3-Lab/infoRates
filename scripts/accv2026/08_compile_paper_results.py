#!/usr/bin/env python3
"""Compile all experimental results for ACCV 2026 paper tables and figures.

Outputs (evaluations/accv2026/paper_results/):
  paper_table_fixed_budget.csv   : All models × all datasets × 4 budgets
  paper_table_tds_metrics.csv    : TDS, AUC, critical_frame_budget per model×dataset
  paper_fig_budget_curves.csv    : Accuracy vs frame budget (for plotting)
  paper_summary.txt              : Human-readable summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "evaluations/accv2026/paper_results"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_BASE = ROOT / "evaluations/accv2026/fixed_budget"

# ─── Run catalog ──────────────────────────────────────────────────────────────
# (model_display, dataset_display, eval_dir, summary_filename)
# SSV2 uses 'somethingv2_validation' prefix; all others use '{dataset}_val'

ALL_RUNS = [
    # ── SSV2 ──────────────────────────────────────────────────────────────────
    ("R3D-18",      "SSv2", "r3d18_ssv2_full_e10_a100",       "somethingv2_validation_accv2026_r3d18_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "SSv2", "mc3_18_ssv2_full_e10_a100",      "somethingv2_validation_accv2026_mc3_18_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "SSv2", "r2plus1d_18_ssv2_full_e10_a100", "somethingv2_validation_accv2026_r2plus1d_18_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","SSv2", "slowfast_r50_ssv2_full_e10_a100","somethingv2_validation_accv2026_slowfast_r50_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "SSv2", "timesformer_ssv2_full_e10_h200",  "somethingv2_validation_accv2026_timesformer_ssv2_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "SSv2", "vivit_ssv2_full_e10_h200",        "somethingv2_validation_accv2026_vivit_ssv2_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "SSv2", "videomae_ssv2_full_e5_h200",      "somethingv2_validation_accv2026_videomae_ssv2_full_e5_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "SSv2", "videomamba_ssv2_full_e10_h200",   "somethingv2_validation_accv2026_videomamba_ssv2_full_e10_h200_fixed_budget_summary.csv"),
    # ── UCF-101 ───────────────────────────────────────────────────────────────
    ("R3D-18",      "UCF-101", "r3d_18_ucf101_full_e10_a100",       "ucf101_val_accv2026_r3d_18_ucf101_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "UCF-101", "mc3_18_ucf101_full_e10_a100",       "ucf101_val_accv2026_mc3_18_ucf101_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "UCF-101", "r2plus1d_18_ucf101_full_e10_a100",  "ucf101_val_accv2026_r2plus1d_18_ucf101_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","UCF-101", "slowfast_r50_ucf101_full_e10_a100", "ucf101_val_accv2026_slowfast_r50_ucf101_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "UCF-101", "timesformer_ucf101_full_e10_h200",  "ucf101_val_accv2026_timesformer_ucf101_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "UCF-101", "vivit_ucf101_full_e10_h200",        "ucf101_val_accv2026_vivit_ucf101_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "UCF-101", "videomae_ucf101_full_e10_h200",     "ucf101_val_accv2026_videomae_ucf101_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "UCF-101", "videomamba_ucf101_full_e10_h200",   "ucf101_val_accv2026_videomamba_ucf101_full_e10_h200_fixed_budget_summary.csv"),
    # ── HMDB-51 ───────────────────────────────────────────────────────────────
    ("R3D-18",      "HMDB-51", "r3d_18_hmdb51_full_e10_a100",       "hmdb51_val_accv2026_r3d_18_hmdb51_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "HMDB-51", "mc3_18_hmdb51_full_e10_a100",       "hmdb51_val_accv2026_mc3_18_hmdb51_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "HMDB-51", "r2plus1d_18_hmdb51_full_e10_a100",  "hmdb51_val_accv2026_r2plus1d_18_hmdb51_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","HMDB-51", "slowfast_r50_hmdb51_full_e10_a100", "hmdb51_val_accv2026_slowfast_r50_hmdb51_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "HMDB-51", "timesformer_hmdb51_full_e10_h200",  "hmdb51_val_accv2026_timesformer_hmdb51_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "HMDB-51", "vivit_hmdb51_full_e10_h200",        "hmdb51_val_accv2026_vivit_hmdb51_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "HMDB-51", "videomae_hmdb51_full_e10_h200",     "hmdb51_val_accv2026_videomae_hmdb51_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "HMDB-51", "videomamba_hmdb51_full_e10_h200",   "hmdb51_val_accv2026_videomamba_hmdb51_full_e10_h200_fixed_budget_summary.csv"),
    # ── DriveAct ──────────────────────────────────────────────────────────────
    ("R3D-18",      "DriveAct", "r3d_18_driveact_full_e10_a100",       "driveact_val_accv2026_r3d_18_driveact_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "DriveAct", "mc3_18_driveact_full_e10_a100",       "driveact_val_accv2026_mc3_18_driveact_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "DriveAct", "r2plus1d_18_driveact_full_e10_a100",  "driveact_val_accv2026_r2plus1d_18_driveact_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","DriveAct", "slowfast_r50_driveact_full_e10_a100", "driveact_val_accv2026_slowfast_r50_driveact_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "DriveAct", "timesformer_driveact_full_e10_h200",  "driveact_val_accv2026_timesformer_driveact_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "DriveAct", "vivit_driveact_full_e10_h200",        "driveact_val_accv2026_vivit_driveact_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "DriveAct", "videomae_driveact_full_e10_h200",     "driveact_val_accv2026_videomae_driveact_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "DriveAct", "videomamba_driveact_full_e10_h200",   "driveact_val_accv2026_videomamba_driveact_full_e10_h200_fixed_budget_summary.csv"),
    # ── Diving-48 ─────────────────────────────────────────────────────────────
    ("R3D-18",      "Diving-48", "r3d_18_diving48_full_e10_a100",       "diving48_val_accv2026_r3d_18_diving48_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "Diving-48", "mc3_18_diving48_full_e10_a100",       "diving48_val_accv2026_mc3_18_diving48_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "Diving-48", "r2plus1d_18_diving48_full_e10_a100",  "diving48_val_accv2026_r2plus1d_18_diving48_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","Diving-48", "slowfast_r50_diving48_full_e10_a100", "diving48_val_accv2026_slowfast_r50_diving48_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "Diving-48", "timesformer_diving48_full_e10_h200",  "diving48_val_accv2026_timesformer_diving48_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "Diving-48", "vivit_diving48_full_e10_h200",        "diving48_val_accv2026_vivit_diving48_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "Diving-48", "videomae_diving48_full_e10_h200",     "diving48_val_accv2026_videomae_diving48_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "Diving-48", "videomamba_diving48_full_e10_h200",   "diving48_val_accv2026_videomamba_diving48_full_e10_h200_fixed_budget_summary.csv"),
    # ── AUTSL ─────────────────────────────────────────────────────────────────
    ("R3D-18",      "AUTSL", "r3d_18_autsl_full_e10_a100",       "autsl_val_accv2026_r3d_18_autsl_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "AUTSL", "mc3_18_autsl_full_e10_a100",       "autsl_val_accv2026_mc3_18_autsl_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "AUTSL", "r2plus1d_18_autsl_full_e10_a100",  "autsl_val_accv2026_r2plus1d_18_autsl_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","AUTSL", "slowfast_r50_autsl_full_e10_a100", "autsl_val_accv2026_slowfast_r50_autsl_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "AUTSL", "timesformer_autsl_full_e10_h200",  "autsl_val_accv2026_timesformer_autsl_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "AUTSL", "vivit_autsl_full_e10_h200",        "autsl_val_accv2026_vivit_autsl_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "AUTSL", "videomae_autsl_full_e10_h200",     "autsl_val_accv2026_videomae_autsl_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "AUTSL", "videomamba_autsl_full_e10_h200",   "autsl_val_accv2026_videomamba_autsl_full_e10_h200_fixed_budget_summary.csv"),
    # ── EPIC-Kitchens ─────────────────────────────────────────────────────────
    ("R3D-18",      "EPIC-Kitchens", "r3d_18_epic_kitchens_full_e10_a100",       "epic_kitchens_val_accv2026_r3d_18_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "EPIC-Kitchens", "mc3_18_epic_kitchens_full_e10_a100",       "epic_kitchens_val_accv2026_mc3_18_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "EPIC-Kitchens", "r2plus1d_18_epic_kitchens_full_e10_a100",  "epic_kitchens_val_accv2026_r2plus1d_18_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast-R50","EPIC-Kitchens", "slowfast_r50_epic_kitchens_full_e10_a100", "epic_kitchens_val_accv2026_slowfast_r50_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "EPIC-Kitchens", "timesformer_epic_kitchens_full_e10_h200",  "epic_kitchens_val_accv2026_timesformer_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "EPIC-Kitchens", "vivit_epic_kitchens_full_e10_h200",        "epic_kitchens_val_accv2026_vivit_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",    "EPIC-Kitchens", "videomae_epic_kitchens_full_e10_h200",     "epic_kitchens_val_accv2026_videomae_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMamba",  "EPIC-Kitchens", "videomamba_epic_kitchens_full_e10_h200",   "epic_kitchens_val_accv2026_videomamba_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"),
    # ── FineGym ───────────────────────────────────────────────────────────────
    ("R3D-18",      "FineGym", "r3d_18_finegym",       "finegym_val_accv2026_r3d_18_finegym_fixed_budget_summary.csv"),
    ("MC3-18",      "FineGym", "mc3_18_finegym",       "finegym_val_accv2026_mc3_18_finegym_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "FineGym", "r2plus1d_18_finegym",  "finegym_val_accv2026_r2plus1d_18_finegym_fixed_budget_summary.csv"),
    ("SlowFast-R50","FineGym", "slowfast_r50_finegym", "finegym_val_accv2026_slowfast_r50_finegym_fixed_budget_summary.csv"),
    ("TimeSformer", "FineGym", "timesformer_finegym",  "finegym_val_accv2026_timesformer_finegym_fixed_budget_summary.csv"),
    ("ViViT",       "FineGym", "vivit_finegym",        "finegym_val_accv2026_vivit_finegym_fixed_budget_summary.csv"),
    ("VideoMAE",    "FineGym", "videomae_finegym",     "finegym_val_accv2026_videomae_finegym_fixed_budget_summary.csv"),
    ("VideoMamba",  "FineGym", "videomamba_finegym",   "finegym_val_accv2026_videomamba_finegym_fixed_budget_summary.csv"),
]


def load_summary(eval_dir: str, summary_file: str):
    p = EVAL_BASE / eval_dir / summary_file
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


def load_temporal_metrics(eval_dir: str):
    p = EVAL_BASE / eval_dir / "temporal_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.iloc[0] if not df.empty else None


# ─── Table: Full fixed-budget results ────────────────────────────────────────

def build_fixed_budget_table():
    rows = []
    acc_col_candidates = ["top1", "top1_accuracy"]
    for model, dataset, eval_dir, summary_file in ALL_RUNS:
        summary = load_summary(eval_dir, summary_file)
        if summary is None:
            print(f"  [MISSING] {model:12s} / {dataset}")
            continue
        acc_col = next((c for c in acc_col_candidates if c in summary.columns), None)
        if acc_col is None:
            continue
        acc = {int(r["budget"]): float(r[acc_col]) for _, r in summary.iterrows()}
        rows.append({
            "Model":   model,
            "Dataset": dataset,
            "Acc@4f":  round(acc.get(4, float("nan")) * 100, 1),
            "Acc@8f":  round(acc.get(8, float("nan")) * 100, 1),
            "Acc@16f": round(acc.get(16, float("nan")) * 100, 1),
            "Acc@32f": round(acc.get(32, float("nan")) * 100, 1),
        })
    return pd.DataFrame(rows)


# ─── Table: TDS / temporal metrics ───────────────────────────────────────────

def build_tds_table():
    rows = []
    for model, dataset, eval_dir, _ in ALL_RUNS:
        tm = load_temporal_metrics(eval_dir)
        if tm is None:
            continue
        rows.append({
            "Model":                model,
            "Dataset":              dataset,
            "Best_Acc":             round(float(tm["best_accuracy"]) * 100, 1),
            "Critical_Budget":      int(tm["critical_frame_budget"]),
            "Temporal_AUC":         round(float(tm["temporal_robustness_auc"]), 4),
        })
    return pd.DataFrame(rows)


# ─── Figure data: budget curves ──────────────────────────────────────────────

def build_budget_curves():
    rows = []
    acc_col_candidates = ["top1", "top1_accuracy"]
    for model, dataset, eval_dir, summary_file in ALL_RUNS:
        summary = load_summary(eval_dir, summary_file)
        if summary is None:
            continue
        acc_col = next((c for c in acc_col_candidates if c in summary.columns), None)
        if acc_col is None:
            continue
        for _, row in summary.iterrows():
            rows.append({
                "model":    model,
                "dataset":  dataset,
                "budget":   int(row["budget"]),
                "accuracy": round(float(row[acc_col]) * 100, 2),
            })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Building fixed-budget results table (all models × all datasets)...")
    t_fixed = build_fixed_budget_table()
    t_fixed.to_csv(OUT / "paper_table_fixed_budget.csv", index=False)
    print(f"  {len(t_fixed)} model×dataset rows")
    print(t_fixed.to_string(index=False))

    print("\nBuilding TDS / temporal metrics table...")
    t_tds = build_tds_table()
    t_tds.to_csv(OUT / "paper_table_tds_metrics.csv", index=False)
    print(f"  {len(t_tds)} rows")
    print(t_tds.to_string(index=False))

    print("\nBuilding budget curve data for figures...")
    curves = build_budget_curves()
    curves.to_csv(OUT / "paper_fig_budget_curves.csv", index=False)
    print(f"  {len(curves)} rows")

    # TDS dataset-level summary (from 07_dataset_temporal_demand.py output)
    tds_path = ROOT / "evaluations/accv2026/dataset_temporal_demand.csv"
    if tds_path.exists():
        tds_df = pd.read_csv(tds_path)
        tds_df.to_csv(OUT / "paper_fig_tds_auc.csv", index=False)
        print(f"\nTDS dataset summary: {len(tds_df)} rows → paper_fig_tds_auc.csv")

    # Human-readable summary
    lines = [
        "ACCV 2026 Paper Results Summary",
        "=" * 60,
        f"Models: 8  |  Datasets: 7  |  Budgets: 4/8/16/32",
        f"Complete runs: {len(t_fixed)} / {len(ALL_RUNS)}",
        "",
        "Fixed-Budget Results:",
        t_fixed.to_string(index=False),
        "",
        "TDS / Temporal Metrics:",
        t_tds.to_string(index=False),
        "",
        "Key findings:",
        "- CNN native frames: R3D-18/MC3-18/R(2+1)D-18/VideoMAE = 16f",
        "- TimeSformer/VideoMamba native: 8f — plateau after 8f budget",
        "- SlowFast-R50/ViViT native: 32f — large jump at 32f budget (architectural)",
        "- AUTSL (sign language): highest temporal demand; VideoMamba fails (feature collapse)",
        "- EPIC-Kitchens: all models ~28-37% at 16f — uniformly hard dataset",
    ]
    (OUT / "paper_summary.txt").write_text("\n".join(lines))
    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
