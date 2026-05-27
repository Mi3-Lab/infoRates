#!/usr/bin/env python3
"""Run fixed-budget ACCV video evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from info_rates.models.model_factory import ModelFactory  # noqa: E402
from info_rates.evaluation.benchmark import evaluate_fixed_budgets, summarize_results  # noqa: E402
from info_rates.models.torchvision_video import load_torchvision_video_checkpoint  # noqa: E402
from info_rates.models.slowfast_video import load_slowfast_checkpoint  # noqa: E402
from info_rates.models.videomamba_model import load_videomamba_checkpoint  # noqa: E402


def load_model_and_processor(args):
    device = args.device
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        print(f"Loading checkpoint: {checkpoint}")
        config_path = checkpoint / "config.json"
        if config_path.exists():
            config_text = config_path.read_text(encoding="utf-8")
            if '"backend": "torchvision_video"' in config_text:
                model, processor, _ = load_torchvision_video_checkpoint(checkpoint, device=device)
                return model, processor
            if '"backend": "slowfast_video"' in config_text:
                model, processor, _ = load_slowfast_checkpoint(checkpoint, device=device)
                return model, processor
            if '"backend": "videomamba"' in config_text:
                model, processor, _ = load_videomamba_checkpoint(checkpoint, device=device)
                model.eval()
                return model, processor
        processor = AutoImageProcessor.from_pretrained(str(checkpoint))
        model = AutoModelForVideoClassification.from_pretrained(str(checkpoint)).to(device)
        model.eval()
        return model, processor

    info = ModelFactory.get_model_info(args.model)
    processor = ModelFactory.load_processor(args.model)
    model, _ = ModelFactory.load_model(
        args.model,
        num_labels=args.num_labels,
        checkpoint=None,
        device=device,
    )
    if args.num_labels != getattr(model.config, "num_labels", args.num_labels):
        print(f"Warning: requested {args.num_labels} labels, model has {model.config.num_labels}")
    return model, processor


def infer_model_frames(args, model) -> int:
    if args.model_frames > 0:
        return args.model_frames
    if args.checkpoint:
        model_type = str(getattr(model.config, "model_type", "")).lower()
        if "videomae" in model_type:
            return 16
        if "timesformer" in model_type:
            return 8
        if "vivit" in model_type:
            return 32
    return int(max(args.budgets)) if args.pad_to_max_budget else 0


def init_wandb_eval(args, output_csv: Path, summary_csv: Path, effective_model_frames: int):
    if args.no_wandb:
        return None, None

    try:
        import wandb
    except ImportError:
        print("[WARN] wandb is not installed; skipping W&B logging.")
        return None, None

    run_name = args.wandb_run_name
    if not run_name:
        model_name = Path(args.checkpoint).name if args.checkpoint else args.model
        run_name = f"fixed-budget-{args.dataset_name or 'dataset'}-{model_name}"

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=args.wandb_tags,
        config={
            "dataset_name": args.dataset_name,
            "split": args.split,
            "model": args.model,
            "checkpoint": args.checkpoint,
            "num_labels": args.num_labels,
            "budgets": args.budgets,
            "model_frames": effective_model_frames,
            "requested_model_frames": args.model_frames,
            "pad_to_max_budget": args.pad_to_max_budget,
            "coverage": args.coverage,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "samples_per_class": args.samples_per_class,
            "resize": args.resize,
            "manifest": args.manifest,
            "output_csv": str(output_csv),
            "summary_csv": str(summary_csv),
            "phase": "eval",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("ACCV_JOB_ID"),
        },
    )
    wandb.log({"status/eval_started": 1})
    return wandb, run


def finish_wandb_eval(
    wandb,
    run,
    summary: pd.DataFrame,
    results: pd.DataFrame,
    output_csv: Path,
    summary_csv: Path,
) -> None:
    if wandb is None or run is None:
        return

    try:
        wandb.log({
            "summary_table": wandb.Table(dataframe=summary),
            "sample_results_table": wandb.Table(dataframe=results),
            "status/eval_finished": 1,
        })
        for _, row in summary.iterrows():
            budget_key = "frame_budget" if "frame_budget" in row else "budget"
            accuracy_key = "accuracy" if "accuracy" in row else "top1"
            budget = int(row[budget_key])
            if accuracy_key in row:
                wandb.log({f"accuracy/budget_{budget}": float(row[accuracy_key])})
            if "top5" in row:
                wandb.log({f"top5/budget_{budget}": float(row["top5"])})
            if "n" in row:
                wandb.log({f"samples/budget_{budget}": int(row["n"])})
        wandb.save(str(output_csv), policy="now")
        wandb.save(str(summary_csv), policy="now")
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model", choices=["timesformer", "videomae", "vivit", "r3d_18", "mc3_18", "r2plus1d_18"], default="timesformer")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-labels", type=int, default=174)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--coverage", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-frames", type=int, default=0)
    parser.add_argument("--pad-to-max-budget", action="store_true")
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset_name or 'dataset'}_{args.split}_{args.checkpoint and Path(args.checkpoint).name or args.model}"
    output_csv = out_dir / f"{stem}_fixed_budget_samples.csv"
    summary_csv = out_dir / f"{stem}_fixed_budget_summary.csv"

    model, processor = load_model_and_processor(args)
    model_frames = infer_model_frames(args, model)
    wandb_module, wandb_run = init_wandb_eval(args, output_csv, summary_csv, model_frames)
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
    finish_wandb_eval(wandb_module, wandb_run, summary, results, output_csv, summary_csv)
    print(f"Wrote sample results: {output_csv}")
    print(f"Wrote summary: {summary_csv}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
