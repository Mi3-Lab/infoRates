#!/usr/bin/env python3
"""Relate temporal-demand scores to fixed-budget recognition behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid["x"].rank().corr(valid["y"].rank()))


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid["x"].corr(valid["y"]))


def class_budget_table(samples: pd.DataFrame, target_fraction: float = 0.95) -> pd.DataFrame:
    valid = samples[~samples["skipped"].astype(bool)].copy()
    grouped = (
        valid.groupby(["dataset", "split", "label_id", "budget"], dropna=False)
        .agg(n=("correct_top1", "size"), top1=("correct_top1", "mean"), top5=("correct_top5", "mean"))
        .reset_index()
    )
    rows = []
    for keys, group in grouped.groupby(["dataset", "split", "label_id"], dropna=False):
        group = group.sort_values("budget")
        best = float(group["top1"].max())
        min_budget = int(group["budget"].min())
        max_budget = int(group["budget"].max())
        sparse_acc = float(group.loc[group["budget"] == min_budget, "top1"].iloc[0])
        dense_acc = float(group.loc[group["budget"] == max_budget, "top1"].iloc[0])
        target = best * target_fraction
        eligible = group[group["top1"] >= target]
        critical = int(eligible["budget"].min()) if not eligible.empty else max_budget
        rows.append(
            {
                "dataset": keys[0],
                "split": keys[1],
                "label_id": int(keys[2]),
                "min_budget": min_budget,
                "max_budget": max_budget,
                "best_accuracy": best,
                "sparse_accuracy": sparse_acc,
                "dense_accuracy": dense_acc,
                "target_accuracy": target,
                "critical_frame_budget": critical,
                "sparse_to_best_drop": best - sparse_acc,
                "sparse_to_dense_drop": dense_acc - sparse_acc,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-csv", required=True)
    parser.add_argument("--samples-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--target-fraction", type=float, default=0.95)
    args = parser.parse_args()

    demand = pd.read_csv(args.demand_csv)
    samples = pd.read_csv(args.samples_csv)
    demand_valid = demand[~demand["skipped"].astype(bool)].copy()

    demand_by_class = (
        demand_valid.groupby(["dataset", "split", "label_id", "label_clean"], dropna=False)
        .agg(
            demand_n=("video_id", "size"),
            mean_demand=("demand_mean_abs_diff", "mean"),
            mean_motion_fraction=("demand_motion_fraction", "mean"),
            p95_demand=("demand_p95_abs_diff", "mean"),
            mean_source_frames=("source_frames", "mean"),
        )
        .reset_index()
    )
    budget_by_class = class_budget_table(samples, target_fraction=args.target_fraction)
    merged = budget_by_class.merge(demand_by_class, on=["dataset", "split", "label_id"], how="left")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    metrics = {
        "n_classes": int(len(merged)),
        "n_classes_with_demand": int(merged["mean_demand"].notna().sum()),
        "target_fraction": float(args.target_fraction),
        "mean_demand_vs_critical_budget_spearman": _rank_corr(merged["mean_demand"], merged["critical_frame_budget"]),
        "mean_demand_vs_sparse_to_best_drop_spearman": _rank_corr(merged["mean_demand"], merged["sparse_to_best_drop"]),
        "motion_fraction_vs_critical_budget_spearman": _rank_corr(
            merged["mean_motion_fraction"], merged["critical_frame_budget"]
        ),
        "mean_demand_vs_critical_budget_pearson": _pearson_corr(merged["mean_demand"], merged["critical_frame_budget"]),
        "mean_demand_vs_sparse_to_best_drop_pearson": _pearson_corr(merged["mean_demand"], merged["sparse_to_best_drop"]),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote merged demand/budget table: {args.output_csv}")
    print(f"Wrote correlation summary: {args.summary_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
