#!/usr/bin/env python3
"""Fixed-budget evaluation for local VideoMamba3 checkpoints."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VM3 = ROOT / "experiments" / "videomamba3"
for path in (SRC, VM3, ROOT / "scripts" / "accv2026"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from info_rates.evaluation.benchmark import evaluate_fixed_budgets, summarize_results  # noqa: E402
from train_videomamba3 import load_videomamba3_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--coverage", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-frames", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--output-dir", default="evaluations/accv2026/fixed_budget")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="inforates-accv2026")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.dataset_name:
        manifest = manifest[manifest["dataset"].astype(str) == args.dataset_name].copy()

    checkpoint = Path(args.checkpoint)
    model, processor, meta = load_videomamba3_checkpoint(checkpoint, device=args.device)
    model_frames = args.model_frames or int(meta.get("num_frames", max(args.budgets)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset_name or 'dataset'}_{args.split}_{checkpoint.name}"
    output_csv = out_dir / f"{stem}_fixed_budget_samples.csv"
    summary_csv = out_dir / f"{stem}_fixed_budget_summary.csv"

    wandb_module = wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or f"fixed-budget-videomamba3-{args.dataset_name or 'dataset'}-{checkpoint.name}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                tags=args.wandb_tags,
                config={
                    "dataset_name": args.dataset_name,
                    "split": args.split,
                    "checkpoint": str(checkpoint),
                    "budgets": args.budgets,
                    "model_frames": model_frames,
                    "resize": args.resize,
                    "manifest": args.manifest,
                    "mamba3_variant": meta.get("mamba3_variant"),
                    "mamba3_impl": meta.get("mamba3_impl"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("ACCV_JOB_ID"),
                },
            )
            wandb_module = wandb
            wandb.log({"status/eval_started": 1})
        except ImportError:
            print("[WARN] wandb is not installed; skipping W&B logging.")

    results = evaluate_fixed_budgets(
        manifest=manifest,
        model=model,
        processor=processor,
        budgets=args.budgets,
        output_csv=output_csv,
        split=args.split,
        coverage=args.coverage,
        stride=args.stride,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        device=args.device,
        resize=args.resize,
        model_frames=model_frames,
        save_logits=args.save_logits,
        logits_dir=out_dir / "logits" / stem,
    )
    summary = summarize_results(results)
    summary.to_csv(summary_csv, index=False)

    if wandb_module is not None and wandb_run is not None:
        try:
            wandb_module.log({
                "summary_table": wandb_module.Table(dataframe=summary),
                "sample_results_table": wandb_module.Table(dataframe=results),
                "status/eval_finished": 1,
            })
            for _, row in summary.iterrows():
                budget_key = "frame_budget" if "frame_budget" in row else "budget"
                accuracy_key = "accuracy" if "accuracy" in row else "top1"
                budget = int(row[budget_key])
                wandb_module.log({f"accuracy/budget_{budget}": float(row[accuracy_key])})
                if "top5" in row:
                    wandb_module.log({f"top5/budget_{budget}": float(row["top5"])})
            wandb_module.save(str(output_csv), policy="now")
            wandb_module.save(str(summary_csv), policy="now")
        finally:
            wandb_run.finish()

    print(f"Wrote sample results: {output_csv}")
    print(f"Wrote summary: {summary_csv}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
