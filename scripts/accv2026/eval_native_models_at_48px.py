#!/usr/bin/env python3
"""Evaluate NATIVE models (trained @ 224px) on 48px resolution WITHOUT retraining.

This tests the degradation when forcing native models to work at 48px.
Compares native @48px vs P3-retrained @48px to quantify retraining benefit.

Usage:
    python eval_native_models_at_48px.py --models vivit videomae mc3_18 r3d_18 --dataset finegym
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from info_rates.evaluation.benchmark import evaluate_fixed_budgets, summarize_results

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

DATASET_CFG = {
    "finegym": dict(manifest="finegym_val_20_per_class.csv", name="finegym", split="val"),
}

SPECIAL_CKPTS = {
    ("slowfast_r50", "finegym"):  "accv2026_slowfast_r50_finegym",
    ("timesformer",  "finegym"):  "accv2026_timesformer_finegym",
    ("vivit",        "finegym"):  "accv2026_vivit_finegym",
    ("videomae",     "finegym"):  "accv2026_videomae_finegym",
    ("videomamba",   "finegym"):  "accv2026_videomamba_finegym",
}

SCRATCH_CKPTS = Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models")


def get_native_checkpoint(model: str, dataset: str) -> Path:
    """Get NATIVE checkpoint (trained at model's native resolution, NOT retrained)."""
    key = (model, dataset)
    candidates = []

    if key in SPECIAL_CKPTS:
        candidates.append(SPECIAL_CKPTS[key])

    # Native resolution (224px or 112px as per MODEL_CFG)
    native_res = MODEL_CFG[model]["resize"]
    suffix = MODEL_CFG[model]["ckpt_suffix"]
    candidates.append(f"accv2026_{model}_{dataset}_224px_e10_{suffix}")
    candidates.append(f"accv2026_{model}_{dataset}_full_e10_{suffix}")
    candidates.append(f"accv2026_{model}_{dataset}")

    for base in [SCRATCH_CKPTS, ROOT / "fine_tuned_models"]:
        for name in candidates:
            p = base / name
            if p.exists():
                return p

    raise FileNotFoundError(f"Native checkpoint not found for {model}/{dataset}. Tried: {candidates}")


def load_model(model_name: str, dataset: str):
    """Load NATIVE model checkpoint."""
    ckpt = get_native_checkpoint(model_name, dataset)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = ckpt / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {ckpt}")
    config_text = config_path.read_text()

    if '"backend": "torchvision_video"' in config_text:
        from info_rates.models.torchvision_video import load_torchvision_video_checkpoint
        model, processor, _ = load_torchvision_video_checkpoint(ckpt, device=device)
    elif '"backend": "slowfast_video"' in config_text:
        from info_rates.models.slowfast_video import load_slowfast_checkpoint
        model, processor, _ = load_slowfast_checkpoint(ckpt, device=device)
    elif '"backend": "videomamba"' in config_text:
        from info_rates.models.videomamba_model import load_videomamba_checkpoint
        model, processor, _ = load_videomamba_checkpoint(str(ckpt), device=device)
    else:
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
        processor = AutoImageProcessor.from_pretrained(str(ckpt))
        model = AutoModelForVideoClassification.from_pretrained(str(ckpt)).to(device)

    model.eval()
    return model, processor, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_CFG),
                        help="Models to evaluate at 48px")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    dcfg = DATASET_CFG[args.dataset]
    manifest_path = ROOT / "evaluations/accv2026/manifests" / dcfg["manifest"]

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    if dcfg["name"]:
        manifest = manifest[manifest["dataset"].astype(str) == dcfg["name"]].copy()
    if manifest.empty:
        print(f"[ERROR] Empty manifest for {args.dataset}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else \
              ROOT / "evaluations/accv2026/native_models_at_48px"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"NATIVE MODELS @ 48px (NO RETRAINING)")
    print(f"{'='*75}\n")

    results_all = []

    for model_name in args.models:
        print(f"\n[MODEL] {model_name}")
        print(f"Loading native checkpoint (trained at native resolution)...")

        try:
            model, processor, device = load_model(model_name, args.dataset)
        except FileNotFoundError as e:
            print(f"  ✗ SKIP: {e}")
            continue

        mcfg = MODEL_CFG[model_name]
        model_frames = mcfg["frames"]
        eval_resolution = 48  # Force to 48px
        coverage = 100  # Standard coverage for comparison
        stride = 1     # Standard stride

        print(f"  native_frames={model_frames}, eval_resolution={eval_resolution}px, coverage={coverage}%, stride={stride}")

        out_csv = out_dir / f"{model_name}_{args.dataset}_48px_samples.csv"
        summary_csv = out_dir / f"{model_name}_{args.dataset}_48px_summary.csv"

        if summary_csv.exists():
            print(f"  [SKIP] Already evaluated")
            df = pd.read_csv(summary_csv)
            if not df.empty:
                row = df.iloc[0]
                results_all.append({
                    "model": model_name,
                    "eval_resolution": eval_resolution,
                    "coverage": coverage,
                    "stride": stride,
                    "native_resolution": mcfg["resize"],
                    "top1": row["top1"],
                    "checkpoint_type": "native",
                })
                print(f"  top1={row['top1']*100:.1f}%")
            continue

        print(f"  Evaluating...", end=" ", flush=True)

        import os
        os.environ["OMP_NUM_THREADS"] = str(args.num_workers)
        os.environ["MKL_NUM_THREADS"] = str(args.num_workers)

        try:
            results = evaluate_fixed_budgets(
                manifest=manifest,
                model=model,
                processor=processor,
                budgets=[model_frames],
                output_csv=out_csv,
                split=dcfg["split"],
                coverage=coverage,
                stride=stride,
                batch_size=args.batch_size,
                device=device,
                resize=eval_resolution,
                model_frames=model_frames,
            )

            summary = summarize_results(results)
            summary.to_csv(summary_csv, index=False)

            if not summary.empty:
                top1 = float(summary.iloc[0]["top1"])
                print(f"top1={top1*100:.1f}%")
                results_all.append({
                    "model": model_name,
                    "eval_resolution": eval_resolution,
                    "coverage": coverage,
                    "stride": stride,
                    "native_resolution": mcfg["resize"],
                    "top1": top1,
                    "checkpoint_type": "native",
                })
            else:
                print("no results")

        except Exception as e:
            print(f"ERROR: {e}")

        del model, processor

    # Summary
    if results_all:
        results_df = pd.DataFrame(results_all)
        results_df.to_csv(out_dir / "native_48px_summary.csv", index=False)

        print(f"\n\n{'='*75}")
        print("SUMMARY: NATIVE MODELS @ 48px")
        print(f"{'='*75}\n")

        for _, row in results_df.iterrows():
            print(f"{row['model']:15s} @ {row['eval_resolution']:3d}px: {row['top1']*100:6.1f}% "
                  f"(native trained @ {row['native_resolution']}px)")

        print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
