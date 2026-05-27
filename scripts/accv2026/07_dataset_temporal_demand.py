#!/usr/bin/env python3
"""Dataset-level Temporal Demand Score (TDS) analysis.

Computes FDE-based TDS for each dataset and correlates with critical_frame_budget.
Shows that dataset-level FDE predicts temporal demand across datasets.
"""

from __future__ import annotations

import argparse
import sys
import concurrent.futures
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def compute_fde(video_path: str, n_probe: int = 5, size: int = 64) -> Optional[float]:
    try:
        import av
        container = av.open(video_path, timeout=10)
        frames_out = []
        container.seek(0)
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="gray")
            h, w = arr.shape
            arr = arr[::max(1, h // size), ::max(1, w // size)][:size, :size].astype(np.float32)
            frames_out.append(arr)
            if len(frames_out) >= n_probe * 4:
                break
        container.close()
        if len(frames_out) < 2:
            return None
        indices = np.linspace(0, len(frames_out) - 1, n_probe, dtype=int)
        frames_out = [frames_out[i] for i in indices]
        diffs = [np.mean(np.abs(frames_out[i+1] - frames_out[i]))
                 for i in range(len(frames_out)-1)]
        return float(np.mean(diffs)) / 255.0
    except Exception:
        return None


def compute_fde_for_dataset(
    samples_csv: str,
    fde_cache: str,
    n_probe: int = 5,
    workers: int = 8,
    max_videos: int = 500,
) -> pd.DataFrame:
    cache_path = Path(fde_cache)
    if cache_path.exists():
        return pd.read_csv(cache_path).dropna(subset=["fde"])

    df = pd.read_csv(samples_csv)
    video_paths = df[["video_id", "video_path"]].drop_duplicates()
    if max_videos > 0:
        video_paths = video_paths.sample(min(len(video_paths), max_videos), random_state=42)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compute_fde, row["video_path"], n_probe): row["video_id"]
                   for _, row in video_paths.iterrows()}
        for fut in concurrent.futures.as_completed(futures):
            vid_id = futures[fut]
            results.append({"video_id": vid_id, "fde": fut.result()})

    fde_df = pd.DataFrame(results).dropna(subset=["fde"])
    fde_df.to_csv(cache_path, index=False)
    return fde_df


DATASETS = {
    "ssv2_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_full_e10_a100/"
                   "somethingv2_validation_accv2026_r2plus1d_18_ssv2_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_full_e10_a100/temporal_metrics.csv",
        "display": "SSV2", "character": "temporal", "model": "R(2+1)D",
    },
    "ssv2_videomae": {
        "samples": "evaluations/accv2026/fixed_budget/videomae_ssv2_full_e5_h200/"
                   "somethingv2_validation_accv2026_videomae_ssv2_full_e5_h200_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/videomae_ssv2_full_e5_h200/temporal_metrics.csv",
        "display": "SSV2", "character": "temporal", "model": "VideoMAE",
    },
    "ucf101_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_ucf101_full_e10_a100/"
                   "ucf101_val_accv2026_r2plus1d_18_ucf101_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_ucf101_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_ucf101_full_e10_a100/temporal_metrics.csv",
        "display": "UCF101", "character": "appearance-biased", "model": "R(2+1)D",
    },
    "hmdb51_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_hmdb51_full_e10_a100/"
                   "hmdb51_val_accv2026_r2plus1d_18_hmdb51_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_hmdb51_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_hmdb51_full_e10_a100/temporal_metrics.csv",
        "display": "HMDB51", "character": "mixed", "model": "R(2+1)D",
    },
    "diving48_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_diving48_full_e10_a100/"
                   "diving48_val_accv2026_r2plus1d_18_diving48_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_diving48_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_diving48_full_e10_a100/temporal_metrics.csv",
        "display": "Diving48", "character": "temporal", "model": "R(2+1)D",
    },
    "ucf101_videomae": {
        "samples": "evaluations/accv2026/fixed_budget/videomae_ucf101_full_e10_h200/"
                   "ucf101_val_accv2026_videomae_ucf101_full_e10_h200_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/videomae_ucf101_full_e10_h200/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/videomae_ucf101_full_e10_h200/temporal_metrics.csv",
        "display": "UCF101", "character": "appearance-biased", "model": "VideoMAE",
    },
    "hmdb51_videomae": {
        "samples": "evaluations/accv2026/fixed_budget/videomae_hmdb51_full_e10_h200/"
                   "hmdb51_val_accv2026_videomae_hmdb51_full_e10_h200_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/videomae_hmdb51_full_e10_h200/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/videomae_hmdb51_full_e10_h200/temporal_metrics.csv",
        "display": "HMDB51", "character": "mixed", "model": "VideoMAE",
    },
    "diving48_videomae": {
        "samples": "evaluations/accv2026/fixed_budget/videomae_diving48_full_e10_h200/"
                   "diving48_val_accv2026_videomae_diving48_full_e10_h200_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/videomae_diving48_full_e10_h200/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/videomae_diving48_full_e10_h200/temporal_metrics.csv",
        "display": "Diving48", "character": "temporal", "model": "VideoMAE",
    },
    "autsl_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_autsl_full_e10_a100/"
                   "autsl_val_accv2026_r2plus1d_18_autsl_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_autsl_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_autsl_full_e10_a100/temporal_metrics.csv",
        "display": "AUTSL", "character": "temporal", "model": "R(2+1)D",
    },
    "driveact_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_driveact_full_e10_a100/"
                   "driveact_val_accv2026_r2plus1d_18_driveact_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_driveact_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_driveact_full_e10_a100/temporal_metrics.csv",
        "display": "DriveAct", "character": "appearance-biased", "model": "R(2+1)D",
    },
    "driveact_videomae": {
        "samples": "evaluations/accv2026/fixed_budget/videomae_driveact_full_e10_h200/"
                   "driveact_val_accv2026_videomae_driveact_full_e10_h200_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/videomae_driveact_full_e10_h200/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/videomae_driveact_full_e10_h200/temporal_metrics.csv",
        "display": "DriveAct", "character": "appearance-biased", "model": "VideoMAE",
    },
    "epic_r2p1d": {
        "samples": "evaluations/accv2026/fixed_budget/r2plus1d_18_epic_kitchens_full_e10_a100/"
                   "epic_kitchens_val_accv2026_r2plus1d_18_epic_kitchens_full_e10_a100_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/r2plus1d_18_epic_kitchens_full_e10_a100/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/r2plus1d_18_epic_kitchens_full_e10_a100/temporal_metrics.csv",
        "display": "EPIC-Kitchens", "character": "temporal", "model": "R(2+1)D",
    },
    "epic_videomae": {
        "samples": "evaluations/accv2026/fixed_budget/videomae_epic_kitchens_full_e10_h200/"
                   "epic_kitchens_val_accv2026_videomae_epic_kitchens_full_e10_h200_fixed_budget_samples.csv",
        "fde_cache": "evaluations/accv2026/fixed_budget/videomae_epic_kitchens_full_e10_h200/fde_cache.csv",
        "temporal_metrics": "evaluations/accv2026/fixed_budget/videomae_epic_kitchens_full_e10_h200/temporal_metrics.csv",
        "display": "EPIC-Kitchens", "character": "temporal", "model": "VideoMAE",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-videos", type=int, default=500,
                        help="Videos to sample per dataset for FDE (0=all)")
    parser.add_argument("--output", default="evaluations/accv2026/dataset_temporal_demand.csv")
    args = parser.parse_args()

    rows = []
    for ds_name, cfg in DATASETS.items():
        if not Path(cfg["samples"]).exists():
            print(f"[SKIP] {ds_name}: samples CSV not found")
            continue
        if not Path(cfg["temporal_metrics"]).exists():
            print(f"[SKIP] {ds_name}: temporal_metrics not found")
            continue

        print(f"\nProcessing {cfg['display']}...")
        fde_df = compute_fde_for_dataset(
            cfg["samples"], cfg["fde_cache"],
            workers=args.workers, max_videos=args.max_videos,
        )
        tds = float(fde_df["fde"].mean())
        tds_std = float(fde_df["fde"].std())
        print(f"  TDS (mean FDE) = {tds:.4f} ± {tds_std:.4f}  (n={len(fde_df)})")

        tm = pd.read_csv(cfg["temporal_metrics"]).iloc[0]
        crit = int(tm["critical_frame_budget"])
        auc = float(tm["temporal_robustness_auc"])
        best_acc = float(tm["best_accuracy"])
        print(f"  critical_frame_budget={crit}  AUC={auc:.4f}  best_acc={best_acc:.4f}")

        rows.append({
            "dataset": ds_name, "display": cfg["display"], "character": cfg["character"],
            "model": cfg.get("model", "unknown"),
            "tds_mean": tds, "tds_std": tds_std, "n_videos": len(fde_df),
            "critical_frame_budget": crit, "temporal_robustness_auc": auc,
            "best_accuracy": best_acc,
        })

    result_df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("DATASET-LEVEL TEMPORAL DEMAND ANALYSIS (R(2+1)D-18)")
    print("="*70)
    print(result_df[["model", "display", "tds_mean", "critical_frame_budget", "temporal_robustness_auc", "best_accuracy"]]
          .sort_values(["model", "tds_mean"]).to_string(index=False))

    # Spearman correlation: TDS vs critical_frame_budget
    from scipy import stats
    r_crit, p_crit = stats.spearmanr(result_df["tds_mean"], result_df["critical_frame_budget"])
    r_auc, p_auc = stats.spearmanr(result_df["tds_mean"], -result_df["temporal_robustness_auc"])
    print(f"\nSpearman r(TDS, critical_budget) = {r_crit:.4f}  p={p_crit:.3f}")
    print(f"Spearman r(TDS, -AUC)           = {r_auc:.4f}  p={p_auc:.3f}")

    result_df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
