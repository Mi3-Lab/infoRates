#!/usr/bin/env python3
"""Coverage × Stride × Resolution sweep for aliasing analysis.

Loads each model once per resolution, then evaluates all 25 (coverage, stride) configs.
Output: one CSV per (model, dataset, resolution, coverage, stride).

Usage:
    python sweep_coverage_stride_resolution.py --model r3d_18 --dataset finegym --resolutions 48 96 112 160 224
    python sweep_coverage_stride_resolution.py --model vivit --dataset finegym --resolutions 96 112 160 224
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

SCRATCH_CKPTS = Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models")


def get_checkpoint(model: str, dataset: str, resolution: int) -> Path:
    """Find checkpoint for model+dataset+resolution.

    Priority:
    1. P3-retrained checkpoint at exact resolution with suffix
    2. Special checkpoints (hardcoded)
    3. Fallback to native resolution
    """
    key = (model, dataset)
    candidates = []

    # P3-retrained checkpoint (PRIORITY)
    suffix = MODEL_CFG[model]["ckpt_suffix"]
    candidates.append(f"accv2026_{model}_{dataset}_{resolution}px_e10_{suffix}")
    candidates.append(f"accv2026_{model}_{dataset}_{resolution}px_e10_h200")  # H200 variant
    candidates.append(f"accv2026_{model}_{dataset}_{resolution}px")

    # Special checkpoints (hardcoded)
    if key in SPECIAL_CKPTS:
        candidates.append(SPECIAL_CKPTS[key])

    # Fallback: native resolution
    candidates.append(f"accv2026_{model}_{dataset}_full_e10_{suffix}")
    candidates.append(f"accv2026_{model}_{dataset}_224px_e10_h200")  # Only if nothing else found

    for base in [SCRATCH_CKPTS, ROOT / "fine_tuned_models"]:
        for name in candidates:
            p = base / name
            if p.exists():
                print(f"    Found: {p.name}")
                return p

    raise FileNotFoundError(f"Checkpoint not found for {model}/{dataset}@{resolution}px. Tried: {candidates}")


def load_model(model_name: str, dataset: str, resolution: int):
    ckpt = get_checkpoint(model_name, dataset, resolution)
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
    parser.add_argument("--model",        required=True, choices=list(MODEL_CFG))
    parser.add_argument("--dataset",      required=True, choices=list(DATASET_CFG))
    parser.add_argument("--resolutions",  nargs="+", type=int, required=True,
                        help="Resolutions to sweep (e.g., 48 96 112 160 224)")
    parser.add_argument("--coverages",    nargs="+", type=int, default=COVERAGES)
    parser.add_argument("--strides",      nargs="+", type=int, default=STRIDES)
    parser.add_argument("--batch-size",   type=int, default=96)  # RTX 6000 Blackwell: 97GB VRAM
    parser.add_argument("--num-workers",  type=int, default=8)   # Ryzen 9950X: 32 cores
    parser.add_argument("--output-dir",   default=None)
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
              ROOT / "evaluations/accv2026/coverage_stride_resolution_sweep" / f"{args.model}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Grid: {len(args.coverages)} coverages × {len(args.strides)} strides × {len(args.resolutions)} resolutions")
    print(f"      = {len(args.coverages) * len(args.strides) * len(args.resolutions)} total evaluations")
    print(f"Output: {out_dir}\n")

    results_all = []
    mcfg = MODEL_CFG[args.model]
    model_frames = mcfg["frames"]

    for resolution in args.resolutions:
        res_dir = out_dir / f"res{resolution}px"
        res_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RES {resolution}px] Loading checkpoint...")
        try:
            model, processor, device = load_model(args.model, args.dataset, resolution)
        except FileNotFoundError as e:
            print(f"  [SKIP] Resolution {resolution}px — {e}")
            continue

        print(f"  model_frames={model_frames}  device={device}")
        print(f"  Running coverage × stride sweep...\n")

        res_results = []

        for coverage in args.coverages:
            for stride in args.strides:
                out_csv = res_dir / f"cov{coverage}_s{stride}_samples.csv"
                summary_csv = res_dir / f"cov{coverage}_s{stride}_summary.csv"

                if summary_csv.exists():
                    df = pd.read_csv(summary_csv)
                    if not df.empty:
                        row = df.iloc[0]
                        res_results.append({
                            "resolution": resolution,
                            "coverage": coverage,
                            "stride": stride,
                            "top1": row["top1"],
                            "n": row["n"]
                        })
                    continue

                print(f"    cov={coverage:3d}%  s={stride:2d} ...", end=" ", flush=True)

                try:
                    # Set environment for better parallelization
                    import os
                    os.environ["OMP_NUM_THREADS"] = str(args.num_workers)
                    os.environ["MKL_NUM_THREADS"] = str(args.num_workers)

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
                        resize=resolution,
                        model_frames=model_frames,
                    )

                    summary = summarize_results(results)
                    summary.to_csv(summary_csv, index=False)

                    if not summary.empty:
                        top1 = float(summary.iloc[0]["top1"])
                        n = int(summary.iloc[0]["n"])
                        print(f"top1={top1*100:.1f}%")
                        res_results.append({
                            "resolution": resolution,
                            "coverage": coverage,
                            "stride": stride,
                            "top1": top1,
                            "n": n
                        })
                    else:
                        print("no results")
                except Exception as e:
                    print(f"ERROR: {e}")

        # Save resolution-specific sweep summary
        if res_results:
            res_df = pd.DataFrame(res_results)
            res_df.to_csv(res_dir / "sweep_summary.csv", index=False)
            results_all.extend(res_results)

            # Pretty pivot for this resolution
            pivot = res_df.pivot(index="coverage", columns="stride", values="top1")
            print(f"\n=== {args.model} / {args.dataset} @ {resolution}px ===")
            print((pivot * 100).round(1).to_string())
            print()

        # Free GPU memory
        del model, processor

    # Save global aggregated table
    if results_all:
        global_df = pd.DataFrame(results_all)
        global_df["model"] = args.model
        global_df["dataset"] = args.dataset
        global_df.to_csv(out_dir / "sweep_summary_all_resolutions.csv", index=False)

        # 3D summary: one heatmap per resolution
        print(f"\n=== GLOBAL SUMMARY: {args.model} / {args.dataset} ===")
        for res in sorted(global_df["resolution"].unique()):
            res_data = global_df[global_df["resolution"] == res]
            pivot = res_data.pivot(index="coverage", columns="stride", values="top1")
            print(f"\nResolution {res}px:")
            print((pivot * 100).round(1).to_string())

    print(f"\n✓ Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
