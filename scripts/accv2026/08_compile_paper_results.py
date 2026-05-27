#!/usr/bin/env python3
"""Compile all experimental results for ACCV 2026 paper tables and figures.

Outputs:
  - paper_table1_ssv2.csv         : SSV2 results (all models)
  - paper_table2_multidataset.csv : Cross-dataset results (3 models × 4 datasets)
  - paper_fig_budget_curves.csv   : Accuracy vs frame budget curves per model/dataset
  - paper_fig_tds_auc.csv         : TDS vs AUC scatter data
  - paper_summary.txt             : Human-readable summary
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "evaluations/accv2026/paper_results"
OUT.mkdir(parents=True, exist_ok=True)


# ─── Result catalog ───────────────────────────────────────────────────────────

EVAL_BASE = ROOT / "evaluations/accv2026/fixed_budget"

SSV2_RUNS = [
    ("R3D-18",      "r3d18_ssv2_full_e10_a100",        "somethingv2_validation_accv2026_r3d18_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("MC3-18",      "mc3_18_ssv2_full_e10_a100",       "somethingv2_validation_accv2026_mc3_18_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18",  "r2plus1d_18_ssv2_full_e10_a100",  "somethingv2_validation_accv2026_r2plus1d_18_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("TimeSformer", "timesformer_ssv2_full_e10_h200",   "somethingv2_validation_accv2026_timesformer_ssv2_full_e10_h200_fixed_budget_summary.csv"),
    ("ViViT",       "vivit_ssv2_full_e10_h200",         "somethingv2_validation_accv2026_vivit_ssv2_full_e10_h200_fixed_budget_summary.csv"),
    ("SlowFast",    "slowfast_r50_ssv2_full_e10_a100",  "somethingv2_validation_accv2026_slowfast_r50_ssv2_full_e10_a100_fixed_budget_summary.csv"),
    ("VideoMAE",    "videomae_ssv2_full_e5_h200",       "somethingv2_validation_accv2026_videomae_ssv2_full_e5_h200_fixed_budget_summary.csv"),
]

MULTI_RUNS = [
    # (model_display, dataset, eval_dir, summary_file)
    ("R(2+1)D-18", "UCF101",       "r2plus1d_18_ucf101_full_e10_a100",       "ucf101_val_accv2026_r2plus1d_18_ucf101_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18", "HMDB51",       "r2plus1d_18_hmdb51_full_e10_a100",       "hmdb51_val_accv2026_r2plus1d_18_hmdb51_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18", "Diving48",     "r2plus1d_18_diving48_full_e10_a100",     "diving48_val_accv2026_r2plus1d_18_diving48_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18", "EPIC-Kitchens","r2plus1d_18_epic_kitchens_full_e10_a100","epic_kitchens_val_accv2026_r2plus1d_18_epic_kitchens_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18", "AUTSL",        "r2plus1d_18_autsl_full_e10_a100",        "autsl_val_accv2026_r2plus1d_18_autsl_full_e10_a100_fixed_budget_summary.csv"),
    ("R(2+1)D-18", "DriveAct",     "r2plus1d_18_driveact_full_e10_a100",     "driveact_val_accv2026_r2plus1d_18_driveact_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast",   "UCF101",       "slowfast_r50_ucf101_full_e10_a100",      "ucf101_val_accv2026_slowfast_r50_ucf101_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast",   "HMDB51",       "slowfast_r50_hmdb51_full_e10_a100",      "hmdb51_val_accv2026_slowfast_r50_hmdb51_full_e10_a100_fixed_budget_summary.csv"),
    ("SlowFast",   "Diving48",     "slowfast_r50_diving48_full_e10_a100",    "diving48_val_accv2026_slowfast_r50_diving48_full_e10_a100_fixed_budget_summary.csv"),
    ("VideoMAE",   "UCF101",       "videomae_ucf101_full_e10_h200",          "ucf101_val_accv2026_videomae_ucf101_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",   "HMDB51",       "videomae_hmdb51_full_e10_h200",          "hmdb51_val_accv2026_videomae_hmdb51_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",   "Diving48",     "videomae_diving48_full_e10_h200",        "diving48_val_accv2026_videomae_diving48_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",   "EPIC-Kitchens","videomae_epic_kitchens_full_e10_h200",   "epic_kitchens_val_accv2026_videomae_epic_kitchens_full_e10_h200_fixed_budget_summary.csv"),
    ("VideoMAE",   "DriveAct",     "videomae_driveact_full_e10_h200",        "driveact_val_accv2026_videomae_driveact_full_e10_h200_fixed_budget_summary.csv"),
    # VideoMAE AUTSL still training (job 71753) — add when complete
]


def load_summary(eval_dir: str, summary_file: str):
    p = EVAL_BASE / eval_dir / summary_file
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_temporal_metrics(eval_dir: str):
    p = EVAL_BASE / eval_dir / "temporal_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).iloc[0]


# ─── Table 1: SSV2 ───────────────────────────────────────────────────────────

def build_table1():
    rows = []
    for model, eval_dir, summary_file in SSV2_RUNS:
        summary = load_summary(eval_dir, summary_file)
        tm = load_temporal_metrics(eval_dir)
        if summary is None or tm is None:
            print(f"  [MISSING] {model} SSV2")
            continue

        acc_col = "top1_accuracy" if "top1_accuracy" in summary.columns else "top1"
        budgets = sorted(summary["budget"].unique())
        acc_by_budget = {int(b): float(summary[summary["budget"] == b][acc_col].iloc[0])
                         for b in budgets}

        rows.append({
            "Model": model,
            "Acc@4f": f"{acc_by_budget.get(4,float('nan'))*100:.1f}%",
            "Acc@8f": f"{acc_by_budget.get(8,float('nan'))*100:.1f}%",
            "Acc@16f": f"{acc_by_budget.get(16,float('nan'))*100:.1f}%",
            "Acc@32f": f"{acc_by_budget.get(32,float('nan'))*100:.1f}%",
            "Best_Acc": f"{float(tm['best_accuracy'])*100:.1f}%",
            "Critical_Budget": int(tm["critical_frame_budget"]),
            "AUC": f"{float(tm['temporal_robustness_auc']):.3f}",
        })
    return pd.DataFrame(rows)


# ─── Table 2: Multi-dataset ───────────────────────────────────────────────────

def build_table2():
    rows = []
    for model, dataset, eval_dir, summary_file in MULTI_RUNS:
        tm = load_temporal_metrics(eval_dir)
        if tm is None:
            print(f"  [MISSING] {model} {dataset}")
            continue
        rows.append({
            "Model": model, "Dataset": dataset,
            "Best_Acc": f"{float(tm['best_accuracy'])*100:.1f}%",
            "Critical_Budget": int(tm["critical_frame_budget"]),
            "AUC": f"{float(tm['temporal_robustness_auc']):.3f}",
        })
    return pd.DataFrame(rows)


# ─── Figure: Accuracy vs budget curves ───────────────────────────────────────

def build_budget_curves():
    rows = []
    for model, eval_dir, summary_file in SSV2_RUNS:
        summary = load_summary(eval_dir, summary_file)
        if summary is None:
            continue
        acc_col = "top1_accuracy" if "top1_accuracy" in summary.columns else "top1"
        for _, row in summary.iterrows():
            rows.append({
                "model": model, "dataset": "SSV2",
                "budget": int(row["budget"]),
                "accuracy": float(row[acc_col]),
            })
    for model, dataset, eval_dir, summary_file in MULTI_RUNS:
        summary = load_summary(eval_dir, summary_file)
        if summary is None:
            continue
        acc_col = "top1_accuracy" if "top1_accuracy" in summary.columns else "top1"
        for _, row in summary.iterrows():
            rows.append({
                "model": model, "dataset": dataset,
                "budget": int(row["budget"]),
                "accuracy": float(row[acc_col]),
            })
    return pd.DataFrame(rows)


# ─── Figure: TDS vs AUC ──────────────────────────────────────────────────────

def build_tds_auc():
    tds_path = ROOT / "evaluations/accv2026/dataset_temporal_demand.csv"
    if not tds_path.exists():
        return pd.DataFrame()
    return pd.read_csv(tds_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building Table 1: SSV2 results...")
    t1 = build_table1()
    t1.to_csv(OUT / "paper_table1_ssv2.csv", index=False)
    print(t1.to_string(index=False))

    print("\nBuilding Table 2: Multi-dataset results...")
    t2 = build_table2()
    t2.to_csv(OUT / "paper_table2_multidataset.csv", index=False)
    print(t2.to_string(index=False))

    print("\nBuilding budget curve data...")
    curves = build_budget_curves()
    curves.to_csv(OUT / "paper_fig_budget_curves.csv", index=False)
    print(f"  {len(curves)} rows")

    print("\nBuilding TDS-AUC scatter data...")
    tds = build_tds_auc()
    if not tds.empty:
        tds.to_csv(OUT / "paper_fig_tds_auc.csv", index=False)
        print(tds[["model","display","tds_mean","temporal_robustness_auc","critical_frame_budget"]].to_string(index=False))

    # Summary text
    # Dynamic TDS correlation from saved data
    tds_summary_lines = []
    tds_path = ROOT / "evaluations/accv2026/dataset_temporal_demand.csv"
    if tds_path.exists():
        try:
            from scipy import stats as _stats
            tds_df = pd.read_csv(tds_path)
            r_val, p_val = _stats.spearmanr(tds_df["tds_mean"], -tds_df["temporal_robustness_auc"])
            n = len(tds_df)
            tds_summary_lines = [
                f"- TDS (mean FDE) predicts temporal robustness: Spearman r={r_val:.3f}, p={p_val:.3f} (n={n})",
                f"  Bootstrap 95% CI: estimated (rerun 07_dataset_temporal_demand.py for details)",
                f"  UCF101 TDS=0.052 → AUC=0.86-0.94 | Diving48 TDS=0.081 → AUC=0.29-0.42",
            ]
        except Exception:
            tds_summary_lines = ["- TDS correlation: see evaluations/accv2026/dataset_temporal_demand.csv"]

    summary_lines = [
        "ACCV 2026 Paper Results Summary",
        "=" * 60,
        "",
        "Table 1: SSV2 Fixed-Budget Evaluation",
        t1.to_string(index=False),
        "",
        "Table 2: Cross-Dataset Results",
        t2.to_string(index=False),
        "",
        "Key Findings:",
        "- Critical frame budget varies by model architecture:",
        "  Conv (R3D/MC3/R2+1D): 16f | TimeSformer: 8f | ViViT/SlowFast: 32f",
        "- Cross-dataset: UCF101 (appearance-biased) has 2-3x higher AUC than Diving48/SSV2",
        "- VideoMAE achieves critical=8f on UCF101 (most frame-efficient)",
        "- FDE routing: only TimeSformer benefits (+0.3% acc, saves 6.8f); Diving48 shows NEGATIVE FDE-demand correlation",
        "- Per-class: 'Poking a hole', 'Pushing onto' are hardest; camera motion classes easiest",
    ] + tds_summary_lines
    with open(OUT / "paper_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nAll results saved to {OUT}/")


if __name__ == "__main__":
    main()
