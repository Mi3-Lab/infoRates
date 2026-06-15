#!/usr/bin/env python3
"""E6 — Spatial Resolution Sweep.

Loads each model ONCE, evaluates at multiple spatial resolutions.
Uses existing fine-tuned checkpoints (no retraining).
Logs each (model, resolution) as a W&B run.

Resolutions chosen relative to native training resolution:
  CNNs (native=112px):       56, 112, 224           (0.5×, 1×, 2×)
  Transformers (native=224px): 112, 224, 336         (0.5×, 1×, 1.5×)
  VideoMamba (native=224px): 112, 224, 336

Usage:
    python sweep_spatial_resolution.py --model r3d_18 --dataset ssv2
    python sweep_spatial_resolution.py --model timesformer --dataset ssv2
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from info_rates.evaluation.benchmark import evaluate_fixed_budgets, summarize_results

# ── Model configs ──────────────────────────────────────────────────────────
# 5-point common grid — all divisible by patch_size=16, valid for ALL architectures
# Mirrors temporal E1 which uses 5 strides: [1, 2, 4, 8, 16]
RESOLUTIONS_ALL = [48, 96, 112, 160, 224]

MODEL_CFG = {
    "r3d_18":       dict(frames=16, native_res=112, ckpt_suffix="a100", resolutions=RESOLUTIONS_ALL),
    "mc3_18":       dict(frames=16, native_res=112, ckpt_suffix="a100", resolutions=RESOLUTIONS_ALL),
    "r2plus1d_18":  dict(frames=16, native_res=112, ckpt_suffix="a100", resolutions=RESOLUTIONS_ALL),
    "slowfast_r50": dict(frames=32, native_res=224, ckpt_suffix="a100", resolutions=RESOLUTIONS_ALL),
    "timesformer":  dict(frames=8,  native_res=224, ckpt_suffix="h200", resolutions=RESOLUTIONS_ALL),
    "vivit":        dict(frames=32, native_res=224, ckpt_suffix="h200", resolutions=RESOLUTIONS_ALL),
    "videomae":     dict(frames=16, native_res=224, ckpt_suffix="h200", resolutions=RESOLUTIONS_ALL),
    "videomamba":   dict(frames=8,  native_res=224, ckpt_suffix="h200", resolutions=RESOLUTIONS_ALL),
}

DATASET_CFG = {
    "ssv2":         dict(manifest="somethingv2_val_20_per_class.csv", name="somethingv2",   split="validation"),
    "ucf101":       dict(manifest="ucf101_val_20_per_class.csv",      name="ucf101",        split="val"),
    "hmdb51":       dict(manifest="hmdb51_val_20_per_class.csv",      name="hmdb51",        split="val"),
    "autsl":        dict(manifest="autsl_val_20_per_class.csv",       name="autsl",         split="val"),
    "diving48":     dict(manifest="diving48_val_20_per_class.csv",    name="diving48",      split="val"),
    "driveact":     dict(manifest="driveact_val_20_per_class.csv",    name="driveact",      split="val"),
    "epic_kitchens":dict(manifest="epic_kitchens_val_20_per_class.csv",name="epic_kitchens",split="val"),
    "flame":        dict(manifest="flame_val_20_per_class.csv",       name="flame",         split="validation"),
    "ufc_crime":    dict(manifest="ufc_crime_val_20_per_class.csv",   name="ufc_crime",     split="validation"),
    "finegym":      dict(manifest="finegym_val_20_per_class.csv",     name="finegym",       split="val"),
}

SPECIAL_CKPTS = {
    ("r3d_18",    "ssv2"): "accv2026_r3d18_ssv2_full_e10_a100",
    ("mc3_18",    "ssv2"): "accv2026_mc3_18_ssv2_full_e10_a100",
    ("r2plus1d_18","ssv2"):"accv2026_r2plus1d_18_ssv2_full_e10_a100",
    ("slowfast_r50","ssv2"):"accv2026_slowfast_r50_ssv2_full_e10_a100",
    ("timesformer","ssv2"):"accv2026_timesformer_ssv2_full_e10_h200",
    ("vivit",     "ssv2"): "accv2026_vivit_ssv2_full_e10_h200",
    ("videomae",  "ssv2"): "accv2026_videomae_ssv2_full_e5_h200",
    ("videomamba","ssv2"): "accv2026_videomamba_ssv2_full_e10_h200",
    # FineGym — checkpoints use short names (no _full_e10_ suffix)
    ("slowfast_r50", "finegym"):  "accv2026_slowfast_r50_finegym",
    ("timesformer",  "finegym"):  "accv2026_timesformer_finegym",
    ("vivit",        "finegym"):  "accv2026_vivit_finegym",
    ("videomae",     "finegym"):  "accv2026_videomae_finegym",
    ("videomamba",   "finegym"):  "accv2026_videomamba_finegym",
}


SCRATCH_CKPTS = Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models")


def get_checkpoint(model: str, dataset: str) -> Path:
    key = (model, dataset)
    candidates = []
    if key in SPECIAL_CKPTS:
        candidates.append(SPECIAL_CKPTS[key])
    # Native-resolution checkpoint first (full_e10): correct for ALL models.
    # For CNNs (native=112px), full_e10_a100 is the 112px model.
    # For Transformers/SlowFast (native=224px), full_e10_h200 is also native.
    # The 224px_e10_h200 checkpoint is a P3-retrain checkpoint: correct only for
    # native-224px models, but WRONG for CNNs (would load a 224px-trained model).
    candidates.append(f"accv2026_{model}_{dataset}_full_e10_{MODEL_CFG[model]['ckpt_suffix']}")
    # 224px retrain as last-resort fallback (only valid when native==224px)
    candidates.append(f"accv2026_{model}_{dataset}_224px_e10_h200")
    for base in [SCRATCH_CKPTS, ROOT / "fine_tuned_models"]:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Checkpoint not found for {model}/{dataset}. Tried: {candidates}")


def load_model(model_name: str, dataset: str):
    ckpt = get_checkpoint(model_name, dataset)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

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
    parser.add_argument("--model",   required=True, choices=list(MODEL_CFG))
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resolutions", default=None,
                        help="Comma-separated resolutions to evaluate (overrides default list)")
    args = parser.parse_args()

    mcfg = MODEL_CFG[args.model]
    dcfg = DATASET_CFG[args.dataset]
    resolutions = [int(r) for r in args.resolutions.split(",")] if args.resolutions else mcfg["resolutions"]
    native_res  = mcfg["native_res"]
    model_frames = mcfg["frames"]

    manifest_path = ROOT / "evaluations/accv2026/manifests" / dcfg["manifest"]
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["dataset"].astype(str) == dcfg["name"]].copy()
    if manifest.empty:
        print(f"[ERROR] Empty manifest for {args.dataset}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else \
              ROOT / "evaluations/accv2026/spatial_resolution_sweep" / f"{args.model}_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} checkpoint for {args.dataset} ...")
    model, processor, device = load_model(args.model, args.dataset)
    print(f"  native_res={native_res}px  frames={model_frames}  device={device}")
    print(f"  resolutions to test: {resolutions}")
    print(f"  manifest: {len(manifest)} rows")
    print()

    # W&B setup — one run per model (log each resolution as a step)
    wandb_enabled = os.environ.get("WANDB_MODE", "online") != "disabled"
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "inforates-accv2026"),
                name=f"e6-spatial-{args.model}-{args.dataset}",
                tags=["accv2026", "e6-spatial-sweep", args.dataset, args.model,
                      "spatial-aliasing", "resolution-ablation"],
                config={
                    "model": args.model,
                    "dataset": args.dataset,
                    "native_res": native_res,
                    "resolutions": resolutions,
                    "model_frames": model_frames,
                    "experiment": "E6_spatial_resolution_sweep",
                },
            )
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}")
            wandb_enabled = False

    results_all = []

    for res in resolutions:
        out_csv     = out_dir / f"res{res}_samples.csv"
        summary_csv = out_dir / f"res{res}_summary.csv"

        if summary_csv.exists():
            print(f"  [SKIP] res={res}px — already done")
            df = pd.read_csv(summary_csv)
            if not df.empty:
                row = df.iloc[0]
                results_all.append({"resolution": res, "top1": row["top1"], "n": row["n"]})
            continue

        label = f"{'NATIVE' if res == native_res else ('sub-native' if res < native_res else 'super-native')}"
        print(f"  Running res={res:3d}px [{label}] ...", end=" ", flush=True)

        results = evaluate_fixed_budgets(
            manifest=manifest,
            model=model,
            processor=processor,
            budgets=[model_frames],
            output_csv=out_csv,
            split=dcfg["split"],
            coverage=100,
            stride=1,
            batch_size=args.batch_size,
            device=device,
            resize=res,
            model_frames=model_frames,
        )

        summary = summarize_results(results)
        summary.to_csv(summary_csv, index=False)

        if not summary.empty:
            top1 = float(summary.iloc[0]["top1"])
            n    = int(summary.iloc[0]["n"])
            drop = (top1 - results_all[0]["top1"]) * 100 if results_all else 0.0
            print(f"top1={top1*100:.1f}%  (Δ={drop:+.1f}pp vs first)  n={n}")
            results_all.append({"resolution": res, "top1": top1, "n": n})

            if wandb_enabled:
                try:
                    import wandb
                    wandb.log({
                        "resolution_px": res,
                        "top1_accuracy": top1,
                        "top1_pct": top1 * 100,
                        "vs_native_pp": (top1 - mcfg.get("_native_top1", top1)) * 100,
                        "is_native": res == native_res,
                    })
                except Exception:
                    pass
        else:
            print("no results")

    # Aggregate — rebuild from ALL res*_summary.csv in dir (merges with existing runs)
    all_rows = []
    import re as _re
    for f in sorted(out_dir.glob("res*_summary.csv")):
        m = _re.match(r"res(\d+)_summary\.csv", f.name)
        if not m:
            continue
        res_val = int(m.group(1))
        try:
            df_r = pd.read_csv(f)
            if not df_r.empty:
                all_rows.append({"resolution": res_val,
                                 "top1": float(df_r.iloc[0]["top1"]),
                                 "n": int(df_r.iloc[0]["n"])})
        except Exception:
            pass
    if all_rows:
        agg = pd.DataFrame(all_rows).sort_values("resolution").reset_index(drop=True)
        agg["model"]      = args.model
        agg["dataset"]    = args.dataset
        agg["native_res"] = native_res
        agg.to_csv(out_dir / "spatial_sweep_summary.csv", index=False)

        native_top1 = agg[agg["resolution"] == native_res]["top1"].values
        print(f"\n=== {args.model} / {args.dataset} — Spatial Resolution Sweep ===")
        print(f"  Native resolution: {native_res}px")
        for _, row in agg.iterrows():
            flag = " ← NATIVE" if row["resolution"] == native_res else ""
            delta = ""
            if len(native_top1) > 0:
                delta = f"  (Δ={( row['top1'] - native_top1[0])*100:+.1f}pp vs native)"
            print(f"  {int(row['resolution']):4d}px → {row['top1']*100:.1f}%{delta}{flag}")

        if wandb_enabled:
            try:
                import wandb
                # Log summary table
                table = wandb.Table(dataframe=agg[["resolution", "top1", "native_res"]])
                wandb.log({"spatial_sweep_table": table})
                wandb.finish()
            except Exception:
                pass

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
