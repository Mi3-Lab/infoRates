#!/usr/bin/env python3
"""Compute per-video temporal-demand scores for ACCV 2026 manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from info_rates.metrics.temporal_demand import compute_temporal_demand, summarize_demand_by_class  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--class-summary-csv", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--frame-budget", type=int, default=16)
    parser.add_argument("--coverage", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize", type=int, default=112)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-threshold", type=float, default=0.08)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    demand = compute_temporal_demand(
        manifest,
        output_csv=args.output_csv,
        split=args.split,
        frame_budget=args.frame_budget,
        coverage=args.coverage,
        stride=args.stride,
        resize=args.resize,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        motion_threshold=args.motion_threshold,
    )
    valid = demand[~demand["skipped"].astype(bool)]
    print(f"Wrote demand scores: {args.output_csv}")
    print(f"Valid videos: {len(valid)} / {len(demand)}")

    if args.class_summary_csv:
        summary = summarize_demand_by_class(demand)
        Path(args.class_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.class_summary_csv, index=False)
        print(f"Wrote class summary: {args.class_summary_csv}")
        if not summary.empty:
            print(summary[["label_id", "n", "mean_demand", "mean_motion_fraction"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
