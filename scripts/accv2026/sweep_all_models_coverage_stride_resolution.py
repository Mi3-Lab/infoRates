#!/usr/bin/env python3
"""Coverage × Stride × Resolution sweep for ALL 8 models.

Parallelizes across models for maximum efficiency.

Usage:
    python sweep_all_models_coverage_stride_resolution.py \
      --dataset finegym \
      --resolutions 48 96 112 160 224 \
      --batch-size 96 \
      --num-workers 8
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import subprocess

ROOT = Path(__file__).resolve().parents[2]

MODEL_CFG = {
    "r3d_18":       dict(frames=16, resize=112, ckpt_suffix="a100"),
    "mc3_18":       dict(frames=16, resize=112, ckpt_suffix="a100"),
    "r2plus1d_18":  dict(frames=16, resize=112, ckpt_suffix="a100"),
    "slowfast_r50": dict(frames=32, resize=224, ckpt_suffix="a100"),
    "timesformer":  dict(frames=8,  resize=224, ckpt_suffix="h200"),
    "vivit":        dict(frames=32, resize=224, ckpt_suffix="h200"),
    "videomae":     dict(frames=16, resize=224, ckpt_suffix="h200"),
    "videomamba":   dict(frames=8,  resize=224, ckpt_suffix="h200"),
}

COVERAGES = [10, 25, 50, 75, 100]
STRIDES = [1, 2, 4, 8, 16]

DATASET_CFG = {
    "finegym": dict(manifest="finegym_val_20_per_class.csv", name="finegym", split="val"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", choices=list(MODEL_CFG),
                        help="Models to sweep (default: all)")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    parser.add_argument("--resolutions", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    models_to_sweep = args.models if args.models else list(MODEL_CFG.keys())

    out_dir = Path(args.output_dir) if args.output_dir else \
              ROOT / "evaluations/accv2026/coverage_stride_resolution_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_configs = len(models_to_sweep) * len(args.resolutions) * len(COVERAGES) * len(STRIDES)

    print(f"\n{'='*80}")
    print(f"COVERAGE × STRIDE × RESOLUTION SWEEP — ALL MODELS")
    print(f"{'='*80}")
    print(f"\nModels:      {len(models_to_sweep)} ({', '.join(models_to_sweep)})")
    print(f"Resolutions: {len(args.resolutions)} ({args.resolutions})")
    print(f"Coverages:   {len(COVERAGES)} ({COVERAGES})")
    print(f"Strides:     {len(STRIDES)} ({STRIDES})")
    print(f"Total:       {total_configs} configurations\n")

    # Run sweep for each model sequentially (each takes ~30-60 min)
    for model in models_to_sweep:
        print(f"\n{'─'*80}")
        print(f"[{model.upper()}] Starting sweep...")
        print(f"{'─'*80}\n")

        cmd = [
            "python", "scripts/accv2026/sweep_coverage_stride_resolution.py",
            "--model", model,
            "--dataset", args.dataset,
            "--resolutions", *[str(r) for r in args.resolutions],
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--output-dir", str(out_dir / f"{model}_{args.dataset}"),
        ]

        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"⚠️  {model} sweep failed with return code {result.returncode}")
        else:
            print(f"✅ {model} sweep completed")

    # Generate global summary
    print(f"\n{'='*80}")
    print(f"Generating global summary...")
    print(f"{'='*80}\n")

    global_results = []
    for model in models_to_sweep:
        model_dir = out_dir / f"{model}_{args.dataset}"
        summary_file = model_dir / "sweep_summary_all_resolutions.csv"

        if summary_file.exists():
            df = pd.read_csv(summary_file)
            global_results.append(df)
            print(f"✓ {model}: {len(df)} results")
        else:
            print(f"✗ {model}: no summary found")

    if global_results:
        global_df = pd.concat(global_results, ignore_index=True)
        global_df.to_csv(out_dir / "GLOBAL_sweep_summary_all_models.csv", index=False)

        print(f"\n✓ Global summary: {len(global_df)} total results")
        print(f"✓ Saved to: {out_dir / 'GLOBAL_sweep_summary_all_models.csv'}")

        # Quick stats
        print(f"\n{'─'*80}")
        print("Quick Stats by Model:")
        print(f"{'─'*80}")
        for model in models_to_sweep:
            model_data = global_df[global_df["model"] == model]
            if not model_data.empty:
                max_acc = model_data["top1"].max() * 100
                mean_acc = model_data["top1"].mean() * 100
                print(f"{model:15s} | max={max_acc:6.1f}% | mean={mean_acc:6.1f}%")


if __name__ == "__main__":
    main()
