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
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from decord import VideoReader, cpu as decord_cpu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from info_rates.evaluation.benchmark import (
    evaluate_fixed_budgets, summarize_results,
    select_frame_indices, adapt_frames_for_model, _move_batch_to_device,
)

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


def get_checkpoint(model: str, dataset: str, train_res: int = None) -> Path:
    """Resolve checkpoint path.

    If train_res is given, loads the resolution-specific retrained checkpoint
    (e.g., accv2026_r3d_18_ucf101_96px_e10_h200) instead of the native one.
    """
    if train_res is not None:
        # Search for best checkpoint: prefer highest val_acc across all versions.
        # Naming: accv2026_{model}_{dataset}_{res}px_e10[_v{N}]_h200
        import glob as _glob
        import json as _json
        transformer_models = {"timesformer", "vivit", "videomae", "videomamba"}
        best_path = None
        best_acc = -1.0

        def _read_val_acc(p: Path) -> float:
            for fname in ("accv_meta.json", "config.json"):
                f = p / fname
                if not f.exists():
                    continue
                try:
                    v = _json.loads(f.read_text()).get("val_acc")
                    if v is not None:
                        return float(v)
                except Exception:
                    pass
            return -1.0

        for base in [SCRATCH_CKPTS, ROOT / "fine_tuned_models"]:
            pattern = str(base / f"accv2026_{model}_{dataset}_{train_res}px_e*_*h200*")
            for p_str in _glob.glob(pattern):
                p = Path(p_str)
                if not p.is_dir():
                    continue
                # Check has valid metadata
                if model in transformer_models:
                    if not (p / "accv_meta.json").exists():
                        continue
                else:
                    cfg = p / "config.json"
                    if not cfg.exists() or '"backend"' not in cfg.read_text():
                        continue
                # Pick by highest val_acc so degraded re-runs (v2 convergence failures)
                # don't override a good v1 checkpoint
                acc = _read_val_acc(p)
                if acc > best_acc:
                    best_acc = acc
                    best_path = p
        if best_path is not None:
            return best_path
        raise FileNotFoundError(
            f"Train-res checkpoint not found for {model}/{dataset}@{train_res}px"
        )

    key = (model, dataset)
    candidates = []
    if key in SPECIAL_CKPTS:
        candidates.append(SPECIAL_CKPTS[key])
    # H200 retrain naming (campanha atual — 224px)
    candidates.append(f"accv2026_{model}_{dataset}_224px_e10_h200")
    # Fallback: nomenclatura antiga
    suffix = MODEL_CFG[model]["ckpt_suffix"]
    candidates.append(f"accv2026_{model}_{dataset}_full_e10_{suffix}")
    for base in [SCRATCH_CKPTS, ROOT / "fine_tuned_models"]:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Checkpoint not found for {model}/{dataset}. Tried: {candidates}")


def load_model(model_name: str, dataset: str, train_res: int = None):
    ckpt = get_checkpoint(model_name, dataset, train_res)
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
        from info_rates.models.model_factory import ModelFactory, _interp_pos_embed
        import json as _json_local
        processor = AutoImageProcessor.from_pretrained(str(ckpt))
        cfg = _json_local.loads(config_text)
        native_size = 224  # all HF video transformers use 224px native
        ckpt_image_size = cfg.get("image_size", native_size) or native_size
        num_labels_ckpt = len(cfg.get("id2label", {})) or cfg.get("num_labels") or 400
        if train_res is not None and ckpt_image_size != native_size:
            # Non-native resolution: PE was NOT saved (plain tensor, not nn.Parameter).
            # Must load base model at 224px (correct PE), interpolate PE to train_res,
            # then overlay fine-tuned weights from checkpoint (PE stays interpolated).
            info = ModelFactory.get_model_info(model_name)
            model = AutoModelForVideoClassification.from_pretrained(
                info["model_id"], num_labels=num_labels_ckpt, ignore_mismatched_sizes=True,
            )
            _interp_pos_embed(model, native_size, train_res)
            model.config.image_size = train_res
            # Update image_size on patch embedding submodules (VideoMAE checks internally)
            for submod in model.modules():
                if submod is model or not hasattr(submod, "image_size"):
                    continue
                old = submod.image_size
                if isinstance(old, (tuple, list)):
                    submod.image_size = (train_res, train_res)
                elif isinstance(old, int) and old == native_size:
                    submod.image_size = train_res
            # Overlay fine-tuned weights (classifier + attention); PE not in safetensors
            from safetensors.torch import load_file as _load_safetensors
            st = next(iter(ckpt.glob("*.safetensors")), None)
            if st:
                state = _load_safetensors(str(st), device="cpu")
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f"  [WARN] PE-fix: {len(missing)} keys missing (expected for PE): {missing[:3]}")
            model = model.to(device)
        else:
            model = AutoModelForVideoClassification.from_pretrained(str(ckpt)).to(device)

    model.eval()
    return model, processor, device


@torch.inference_mode()
def run_sweep_fast(
    manifest_df: pd.DataFrame,
    model,
    processor,
    coverages: list,
    strides: list,
    model_frames: int,
    resize: int,
    device: str,
    batch_size: int = 32,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Decode each video ONCE, run all configs from frame cache.

    25x less disk I/O vs the config-outer approach; GPU stays busy.
    """
    configs = [(c, s) for c in coverages for s in strides]
    num_labels = int(getattr(model.config, "num_labels",
                             len(getattr(model.config, "id2label", {}))))
    device_obj = torch.device(device)

    df = manifest_df.copy()
    if "exists" in df.columns:
        df = df[df["exists"].astype(bool)]
    if "split" in df.columns and hasattr(df, "_split_filter"):
        pass  # already filtered upstream
    df = df.reset_index(drop=True)

    # accumulators: per config → list of correct booleans
    correct: dict = {cfg: [] for cfg in configs}

    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    t0_total = time.perf_counter()

    for chunk_idx in range(n_chunks):
        chunk = df.iloc[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]

        # ── 1. Decode every video in the chunk ONCE ──────────────────────
        cache: list = []  # (frames_dict, total_frames, label_id) or None
        for row in chunk.itertuples(index=False):
            row_d = row._asdict()
            label_id = int(row_d["label_id"])
            if label_id < 0 or label_id >= num_labels:
                cache.append(None)
                continue
            try:
                vr = VideoReader(str(row_d["video_path"]), ctx=decord_cpu(0))
                total = len(vr)
                # Compute union of all frame positions needed across all 25 configs
                needed = set()
                for (cov, s) in configs:
                    needed.update(select_frame_indices(total, model_frames, cov, s).tolist())
                needed_sorted = sorted(needed)
                raw = vr.get_batch(needed_sorted).asnumpy()  # [K, H, W, 3]
                resized = np.stack([cv2.resize(f, (resize, resize)) for f in raw])
                pos_map = {pos: i for i, pos in enumerate(needed_sorted)}
                cache.append((resized, pos_map, total, label_id, row_d))
            except Exception:
                cache.append(None)

        # ── 2. For each config, select frames from cache → batch GPU ──────
        for (cov, s) in configs:
            batch_frames: list = []
            batch_labels: list = []

            def _flush(bf, bl):
                if not bf:
                    return
                inp = _move_batch_to_device(
                    processor(bf, return_tensors="pt"), device_obj)
                with torch.amp.autocast(device_type=device_obj.type,
                                        enabled=device_obj.type == "cuda"):
                    logits = model(**inp).logits
                preds = logits.argmax(dim=-1).cpu().numpy()
                for pred, lbl in zip(preds, bl):
                    correct[(cov, s)].append(int(pred) == lbl)
                del inp, logits

            for entry in cache:
                if entry is None:
                    continue
                resized, pos_map, total, label_id, _ = entry
                idx = select_frame_indices(total, model_frames, cov, s)
                frames = [resized[pos_map[min(i, max(pos_map))]] for i in idx]
                frames = adapt_frames_for_model(frames, model_frames)
                batch_frames.append(frames)
                batch_labels.append(label_id)
                if len(batch_frames) >= batch_size:
                    _flush(batch_frames, batch_labels)
                    batch_frames, batch_labels = [], []
            _flush(batch_frames, batch_labels)

        elapsed = time.perf_counter() - t0_total
        done_vids = min((chunk_idx + 1) * chunk_size, len(df))
        print(f"  chunk {chunk_idx+1}/{n_chunks}  ({done_vids}/{len(df)} videos"
              f"  {elapsed:.0f}s)", flush=True)

    rows = []
    for (cov, s) in configs:
        c = correct[(cov, s)]
        n = len(c)
        rows.append({"coverage": cov, "stride": s,
                     "top1": sum(c) / max(n, 1), "n": n})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list(MODEL_CFG))
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    parser.add_argument("--coverages",   nargs="+", type=int, default=COVERAGES)
    parser.add_argument("--strides",     nargs="+", type=int, default=STRIDES)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--input-size",  type=int, default=None,
                        help="Override spatial resolution for inference (default: model's native)")
    parser.add_argument("--train-res",   type=int, default=None,
                        help="Load resolution-specific retrained checkpoint (e.g. 96, 112, 160, 224)")
    parser.add_argument("--no-fast", action="store_true",
                        help="Disable fast cached sweep (use legacy per-config decode)")
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

    # Resolution for inference: explicit --input-size > --train-res > model native
    if args.input_size:
        resize = args.input_size
    elif args.train_res:
        resize = args.train_res
    else:
        resize = mcfg["resize"]

    # Output directory tag
    if args.train_res:
        res_tag = f"_trainres{args.train_res}"
    elif args.input_size and args.input_size != mcfg["resize"]:
        res_tag = f"_res{resize}"
    else:
        res_tag = ""

    out_dir = Path(args.output_dir) if args.output_dir else \
              ROOT / "evaluations/accv2026/coverage_stride_sweep" / f"{args.model}_{args.dataset}{res_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} checkpoint for {args.dataset} (train_res={args.train_res})...")
    model, processor, device = load_model(args.model, args.dataset, args.train_res)
    model_frames = mcfg["frames"]
    print(f"  model_frames={model_frames}  resize={resize}  device={device}")
    print(f"  manifest: {len(manifest)} rows")
    print(f"  configs: {len(args.coverages)} coverages × {len(args.strides)} strides = {len(args.coverages)*len(args.strides)}")
    print()

    results_all = []

    # Check which configs are already done
    pending_coverages, pending_strides = [], []
    for coverage in args.coverages:
        for stride in args.strides:
            summary_csv = out_dir / f"cov{coverage}_s{stride}_summary.csv"
            if summary_csv.exists():
                print(f"  [SKIP] cov={coverage}% s={stride} — already done")
                df_s = pd.read_csv(summary_csv)
                if not df_s.empty:
                    row = df_s.iloc[0]
                    results_all.append({"coverage": coverage, "stride": stride,
                                        "top1": row["top1"], "n": row["n"]})
            else:
                pending_coverages.append(coverage)
                pending_strides.append(stride)

    pending_configs = list(dict.fromkeys(zip(pending_coverages, pending_strides)))  # unique ordered

    if pending_configs and not args.no_fast:
        # Fast path: decode each video once, run all pending configs from cache
        covs  = sorted(set(c for c, _ in pending_configs))
        strs  = sorted(set(s for _, s in pending_configs))
        print(f"\n  Fast sweep: {len(pending_configs)} configs × {len(manifest)} videos (decode-once)")
        split_manifest = manifest[manifest["split"].astype(str) == dcfg["split"]].copy() \
            if "split" in manifest.columns and dcfg["split"] else manifest.copy()
        if "exists" in split_manifest.columns:
            split_manifest = split_manifest[split_manifest["exists"].astype(bool)]

        fast_df = run_sweep_fast(
            manifest_df=split_manifest,
            model=model,
            processor=processor,
            coverages=covs,
            strides=strs,
            model_frames=model_frames,
            resize=resize,
            device=device,
            batch_size=args.batch_size,
        )
        for _, row in fast_df.iterrows():
            coverage, stride = int(row["coverage"]), int(row["stride"])
            if (coverage, stride) not in {(c, s) for c, s in pending_configs}:
                continue
            top1, n = float(row["top1"]), int(row["n"])
            # Write per-config summary for resume compatibility
            summary_csv = out_dir / f"cov{coverage}_s{stride}_summary.csv"
            pd.DataFrame([{"dataset": dcfg["name"], "split": dcfg["split"],
                           "budget": model_frames, "coverage": coverage,
                           "stride": stride, "top1": top1, "n": n,
                           "top5": top1, "mean_decode_s": 0.0,
                           "mean_inference_s": 0.0}]).to_csv(summary_csv, index=False)
            print(f"  cov={coverage:3d}%  s={stride:2d}  top1={top1*100:.1f}%  n={n}")
            results_all.append({"coverage": coverage, "stride": stride,
                                 "top1": top1, "n": n})

    elif pending_configs:
        # Legacy path (--no-fast flag)
        for coverage, stride in pending_configs:
            out_csv = out_dir / f"cov{coverage}_s{stride}_samples.csv"
            summary_csv = out_dir / f"cov{coverage}_s{stride}_summary.csv"
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
