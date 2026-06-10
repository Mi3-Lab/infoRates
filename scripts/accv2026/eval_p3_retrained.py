#!/usr/bin/env python3
"""Evaluate P3-retrained checkpoints at their training resolution.

For each checkpoint accv2026_{model}_{dataset}_{res}px_e10_*, evaluates
the model AT that resolution and saves to evaluations/accv2026/p3_retrained/.

This complements sweep_spatial_resolution.py (which uses native checkpoints).
Together they answer: "Does retraining at resolution X close the accuracy gap?"

Usage:
    python scripts/accv2026/eval_p3_retrained.py --dataset finegym
    python scripts/accv2026/eval_p3_retrained.py --dataset finegym --model r3d_18
    python scripts/accv2026/eval_p3_retrained.py  # all datasets
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from info_rates.evaluation.benchmark import evaluate_fixed_budgets, summarize_results

RESOLUTIONS = [96, 112, 160, 224]

DATASET_CFG = {
    "ssv2":          dict(manifest="somethingv2_val_20_per_class.csv",  name="somethingv2",    split="validation"),
    "ucf101":        dict(manifest="ucf101_val_20_per_class.csv",       name="ucf101",         split="val"),
    "hmdb51":        dict(manifest="hmdb51_val_20_per_class.csv",       name="hmdb51",         split="val"),
    "autsl":         dict(manifest="autsl_val_20_per_class.csv",        name="autsl",          split="val"),
    "diving48":      dict(manifest="diving48_val_20_per_class.csv",     name="diving48",       split="val"),
    "driveact":      dict(manifest="driveact_val_20_per_class.csv",     name="driveact",       split="val"),
    "epic_kitchens": dict(manifest="epic_kitchens_val_20_per_class.csv",name="epic_kitchens",  split="val"),
    "finegym":       dict(manifest="finegym_val_20_per_class.csv",      name="finegym",        split="val"),
}

MODEL_FRAMES = {
    "r3d_18": 16, "mc3_18": 16, "r2plus1d_18": 16,
    "slowfast_r50": 32,
    "timesformer": 8, "vivit": 32, "videomae": 16, "videomamba": 8,
}

# Search order: local fine_tuned_models/, then scratch
CKPT_BASES = [
    ROOT / "fine_tuned_models",
    Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"),
]


def find_p3_checkpoints(dataset: str | None = None, model: str | None = None) -> list[dict]:
    """Find all P3-retrained checkpoints matching the filters."""
    found = []
    for base in CKPT_BASES:
        if not base.exists():
            continue
        pattern = "accv2026_*_*_*px_e10_*"
        for ckpt in sorted(base.glob(pattern)):
            if not (ckpt / "config.json").exists():
                continue
            # Parse: accv2026_{model}_{dataset}_{res}px_e{epochs}_{suffix}
            parts = ckpt.name.split("_")
            # Find the {res}px part
            res_idx = next((i for i, p in enumerate(parts) if p.endswith("px")), None)
            if res_idx is None:
                continue
            res_str = parts[res_idx].replace("px", "")
            if not res_str.isdigit():
                continue
            res = int(res_str)
            if res not in RESOLUTIONS:
                continue
            # model = parts[1..res_idx-2], dataset = parts[res_idx-1]
            ds = parts[res_idx - 1]
            mdl = "_".join(parts[1:res_idx - 1])

            if dataset and ds != dataset:
                continue
            if model and mdl != model:
                continue
            if mdl not in MODEL_FRAMES:
                continue
            if ds not in DATASET_CFG:
                continue

            found.append({"model": mdl, "dataset": ds, "resolution": res, "ckpt": ckpt})
    return found


def load_model(ckpt: Path, model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_text = (ckpt / "config.json").read_text()

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
    parser.add_argument("--dataset", default=None, choices=list(DATASET_CFG) + [None])
    parser.add_argument("--model",   default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    jobs = find_p3_checkpoints(args.dataset, args.model)
    if not jobs:
        print(f"No P3-retrained checkpoints found for dataset={args.dataset}, model={args.model}")
        print("Run run_p3_retrain_finegym.sh first.")
        sys.exit(1)

    print(f"Found {len(jobs)} P3-retrained checkpoints to evaluate")
    print()

    out_root = Path(args.output_dir) if args.output_dir else \
               ROOT / "evaluations/accv2026/p3_retrained"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for job in jobs:
        mdl, ds, res, ckpt = job["model"], job["dataset"], job["resolution"], job["ckpt"]
        dcfg = DATASET_CFG[ds]
        frames = MODEL_FRAMES[mdl]

        out_dir = out_root / f"{mdl}_{ds}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / f"res{res}_retrained_summary.csv"

        if summary_csv.exists():
            print(f"  [SKIP] {mdl}/{ds}@{res}px retrained — already evaluated")
            df = pd.read_csv(summary_csv)
            if not df.empty:
                row = df.iloc[0]
                summary_rows.append({
                    "model": mdl, "dataset": ds, "resolution": res,
                    "top1": row["top1"], "n": row["n"],
                    "ckpt": ckpt.name,
                })
            continue

        manifest_path = ROOT / "evaluations/accv2026/manifests" / dcfg["manifest"]
        if not manifest_path.exists():
            print(f"  [SKIP] {mdl}/{ds} — manifest not found: {manifest_path}")
            continue

        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["dataset"].astype(str) == dcfg["name"]].copy()
        if manifest.empty:
            print(f"  [SKIP] {mdl}/{ds} — empty manifest")
            continue

        print(f"  Evaluating {mdl}/{ds}@{res}px (retrained) ...")
        try:
            model_obj, processor, device = load_model(ckpt, mdl)
        except Exception as e:
            print(f"  [ERROR] load {ckpt}: {e}")
            continue

        results = evaluate_fixed_budgets(
            manifest=manifest,
            model=model_obj,
            processor=processor,
            budgets=[frames],
            output_csv=out_dir / f"res{res}_retrained_samples.csv",
            split=dcfg["split"],
            coverage=100,
            stride=1,
            batch_size=args.batch_size,
            device=device,
            resize=res,
            model_frames=frames,
        )
        summary = summarize_results(results)
        summary.to_csv(summary_csv, index=False)

        if not summary.empty:
            top1 = float(summary.iloc[0]["top1"])
            n = int(summary.iloc[0]["n"])
            print(f"    → top1={top1*100:.1f}%  n={n}")
            summary_rows.append({
                "model": mdl, "dataset": ds, "resolution": res,
                "top1": top1, "n": n, "ckpt": ckpt.name,
            })

        del model_obj
        torch.cuda.empty_cache()

    if summary_rows:
        agg = pd.DataFrame(summary_rows)
        agg.to_csv(out_root / "p3_retrained_summary.csv", index=False)
        print(f"\nSaved {len(agg)} results to {out_root}/p3_retrained_summary.csv")

        # Compare vs native evaluation (from p3_results.csv)
        native_csv = ROOT / "dashboard/data/p3_results.csv"
        if native_csv.exists():
            native = pd.read_csv(native_csv)
            print("\n=== P3 Retrain vs Native Eval (same resolution) ===")
            print(f"{'Model':<15} {'Dataset':<12} {'Res':>4}  {'Native':>7}  {'Retrained':>9}  {'Δ':>7}")
            print("-" * 60)
            for _, row in agg.iterrows():
                nat_row = native[
                    (native["model"] == row["model"]) &
                    (native["dataset"] == row["dataset"]) &
                    (native["res"] == row["resolution"])
                ]
                nat_acc = nat_row["acc"].values[0] * 100 if not nat_row.empty else float("nan")
                ret_acc = row["top1"] * 100
                delta = ret_acc - nat_acc
                print(f"  {row['model']:<13} {row['dataset']:<12} {row['resolution']:>4}px"
                      f"  {nat_acc:>6.1f}%  {ret_acc:>8.1f}%  {delta:>+6.1f}pp")


if __name__ == "__main__":
    main()
