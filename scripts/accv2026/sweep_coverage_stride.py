#!/usr/bin/env python3
"""Coverage × Stride sweep for aliasing analysis.

Loads each model ONCE, then evaluates all 25 (coverage, stride) configs.
Output: one CSV per (model, dataset, coverage, stride).

Usage:
    python sweep_coverage_stride.py --model r3d_18 --dataset ucf101
    python sweep_coverage_stride.py --model videomae --dataset autsl
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from info_rates.evaluation.benchmark import evaluate_fixed_budgets, summarize_results

# ── Sweep grid (matches ECCV paper protocol) ──────────────────────────────
COVERAGES = [10, 25, 50, 75, 100]
STRIDES   = [1, 2, 4, 8, 16]

# ── Model configs ─────────────────────────────────────────────────────────
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

# ── Dataset manifests and splits ──────────────────────────────────────────
DATASET_CFG = {
    "ucf101":        dict(manifest="ucf101_val_20_per_class.csv",       name="ucf101",       split="val"),
    "ssv2":          dict(manifest="somethingv2_val_20_per_class.csv",   name="somethingv2",  split="validation"),
    "hmdb51":        dict(manifest="hmdb51_val_20_per_class.csv",        name="hmdb51",       split="val"),
    "diving48":      dict(manifest="diving48_val_20_per_class.csv",      name="diving48",     split="val"),
    "autsl":         dict(manifest="autsl_val_20_per_class.csv",         name="autsl",        split="val"),
    "driveact":      dict(manifest="driveact_val_20_per_class.csv",      name="driveact",     split="val"),
    "epic_kitchens": dict(manifest="epic_kitchens_val_20_per_class.csv", name="epic_kitchens",split="val"),
    "finegym":       dict(manifest="finegym_val_20_per_class.csv",       name="finegym",      split="val"),
}

# ── Special checkpoint names ──────────────────────────────────────────────
SPECIAL_CKPTS = {
    ("r3d_18",    "ssv2"): "accv2026_r3d18_ssv2_full_e10_a100",
    ("mc3_18",    "ssv2"): "accv2026_mc3_18_ssv2_full_e10_a100",
    ("r2plus1d_18","ssv2"):"accv2026_r2plus1d_18_ssv2_full_e10_a100",
    ("slowfast_r50","ssv2"):"accv2026_slowfast_r50_ssv2_full_e10_a100",
    ("timesformer","ssv2"):"accv2026_timesformer_ssv2_full_e10_h200",
    ("vivit",     "ssv2"): "accv2026_vivit_ssv2_full_e10_h200",
    ("videomae",  "ssv2"): "accv2026_videomae_ssv2_full_e5_h200",
    ("videomamba","ssv2"): "accv2026_videomamba_ssv2_full_e10_h200",
    # FineGym — checkpoints saved with short names (no _full_e10_ suffix)
    ("slowfast_r50", "finegym"):  "accv2026_slowfast_r50_finegym",
    ("timesformer",  "finegym"):  "accv2026_timesformer_finegym",
    ("vivit",        "finegym"):  "accv2026_vivit_finegym",
    ("videomae",     "finegym"):  "accv2026_videomae_finegym",
    ("videomamba",   "finegym"):  "accv2026_videomamba_finegym",
}


def get_checkpoint(model: str, dataset: str) -> Path:
    key = (model, dataset)
    if key in SPECIAL_CKPTS:
        name = SPECIAL_CKPTS[key]
    else:
        suffix = MODEL_CFG[model]["ckpt_suffix"]
        name = f"accv2026_{model}_{dataset}_full_e10_{suffix}"
    return ROOT / "fine_tuned_models" / name


def load_model(model_name: str, dataset: str):
    ckpt = get_checkpoint(model_name, dataset)
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
        # HuggingFace transformer (TimeSformer, ViViT, VideoMAE)
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
        processor = AutoImageProcessor.from_pretrained(str(ckpt))
        model = AutoModelForVideoClassification.from_pretrained(str(ckpt)).to(device)

    model.eval()
    return model, processor, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list(MODEL_CFG))
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    parser.add_argument("--coverages", nargs="+", type=int, default=COVERAGES)
    parser.add_argument("--strides",   nargs="+", type=int, default=STRIDES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    mcfg = MODEL_CFG[args.model]
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
              ROOT / "evaluations/accv2026/coverage_stride_sweep" / f"{args.model}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} checkpoint for {args.dataset}...")
    model, processor, device = load_model(args.model, args.dataset)
    model_frames = mcfg["frames"]
    resize = mcfg["resize"]
    print(f"  model_frames={model_frames}  resize={resize}  device={device}")
    print(f"  manifest: {len(manifest)} rows")
    print(f"  configs: {len(args.coverages)} coverages × {len(args.strides)} strides = {len(args.coverages)*len(args.strides)}")
    print()

    results_all = []

    for coverage in args.coverages:
        for stride in args.strides:
            out_csv = out_dir / f"cov{coverage}_s{stride}_samples.csv"
            summary_csv = out_dir / f"cov{coverage}_s{stride}_summary.csv"

            if summary_csv.exists():
                print(f"  [SKIP] cov={coverage}% s={stride} — already done")
                df = pd.read_csv(summary_csv)
                if not df.empty:
                    row = df.iloc[0]
                    results_all.append({"coverage": coverage, "stride": stride,
                                        "top1": row["top1"], "n": row["n"]})
                continue

            print(f"  Running cov={coverage:3d}%  s={stride:2d} ...", end=" ", flush=True)

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
                resize=resize,
                model_frames=model_frames,
            )

            summary = summarize_results(results)
            summary.to_csv(summary_csv, index=False)

            if not summary.empty:
                top1 = float(summary.iloc[0]["top1"])
                n    = int(summary.iloc[0]["n"])
                print(f"top1={top1*100:.1f}%  n={n}")
                results_all.append({"coverage": coverage, "stride": stride,
                                    "top1": top1, "n": n})
            else:
                print("no results")

    # Save aggregated table: coverage × stride → top1
    if results_all:
        agg = pd.DataFrame(results_all)
        agg["model"]   = args.model
        agg["dataset"] = args.dataset
        agg.to_csv(out_dir / "sweep_summary.csv", index=False)

        # Pretty pivot
        pivot = agg.pivot(index="coverage", columns="stride", values="top1")
        print(f"\n=== {args.model} / {args.dataset} — top1 by coverage×stride ===")
        print((pivot * 100).round(1).to_string())

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
