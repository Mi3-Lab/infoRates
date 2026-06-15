#!/usr/bin/env python3
"""Inference latency benchmark: all 8 models × 5 resolutions.

Loads each model from its FineGym P3-retrained checkpoint (which already exists
at every resolution), creates synthetic (random) input tensors, and measures
wall-clock forward-pass latency via CUDA events.

Warmup: 20 iterations.   Benchmark: 100 iterations.   Batch size: 1.
GPU sync is enforced between each measurement.

Output:
    evaluations/accv2026/paper_results/latency_by_resolution.csv
    Columns: model, resolution, mean_ms, std_ms, gpu
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CKPT_ROOT    = ROOT / "fine_tuned_models"
OUT_CSV      = ROOT / "evaluations/accv2026/paper_results/latency_by_resolution.csv"
RESOLUTIONS  = [48, 96, 112, 160, 224]
WARMUP_ITERS = 20
BENCH_ITERS  = 100

MODEL_FRAMES = {
    "r3d_18":       16,
    "mc3_18":       16,
    "r2plus1d_18":  16,
    "slowfast_r50": 32,   # 8 slow + 32 fast (handled separately)
    "timesformer":  8,
    "vivit":        32,
    "videomae":     16,
    "videomamba":   8,
}


def ckpt_path(model: str, res: int) -> Path:
    """Resolve FineGym P3 checkpoint for (model, resolution)."""
    candidates = [
        CKPT_ROOT / f"accv2026_{model}_finegym_{res}px_e10_h200",
        CKPT_ROOT / f"accv2026_{model}_finegym_{res}px_e10_a100",
        CKPT_ROOT / f"accv2026_{model}_finegym_{res}px",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint for {model}@{res}px. Tried: {[str(c) for c in candidates]}")


def backend(ckpt: Path) -> str:
    cfg = (ckpt / "config.json").read_text()
    if '"backend": "torchvision_video"' in cfg:
        return "torchvision"
    if '"backend": "slowfast_video"' in cfg:
        return "slowfast"
    if '"backend": "videomamba"' in cfg:
        return "videomamba"
    return "huggingface"


def load(model_name: str, ckpt: Path, device: str):
    b = backend(ckpt)
    if b == "torchvision":
        from info_rates.models.torchvision_video import load_torchvision_video_checkpoint
        model, _, _ = load_torchvision_video_checkpoint(ckpt, device=device)
    elif b == "slowfast":
        from info_rates.models.slowfast_video import load_slowfast_checkpoint
        model, _, _ = load_slowfast_checkpoint(ckpt, device=device)
    elif b == "videomamba":
        from info_rates.models.videomamba_model import load_videomamba_checkpoint
        model, _, _ = load_videomamba_checkpoint(str(ckpt), device=device)
    else:
        from transformers import AutoModelForVideoClassification
        model = AutoModelForVideoClassification.from_pretrained(str(ckpt)).to(device)
    model.eval()
    return model, b


def make_input(b: str, model_name: str, res: int, device: str):
    """Create a synthetic input matching each model's expected tensor format."""
    T = MODEL_FRAMES[model_name]
    if b == "torchvision":
        # TorchVision video: (B, C, T, H, W)
        return {"pixel_values": torch.randn(1, 3, T, res, res, device=device)}
    if b == "slowfast":
        # SlowFast: separate slow (8f) + fast (32f) pathways
        return {
            "slow_frames": torch.randn(1, 3, 8,  res, res, device=device),
            "fast_frames": torch.randn(1, 3, 32, res, res, device=device),
        }
    if b == "videomamba":
        # VideoMamba processor outputs (B, C, T, H, W)
        return {"pixel_values": torch.randn(1, 3, T, res, res, device=device)}
    # HuggingFace transformers: (B, T, 3, H, W)  [time axis first]
    return {"pixel_values": torch.randn(1, T, 3, res, res, device=device)}


def measure(model, inputs: dict, device: str) -> tuple[float, float]:
    """Return (mean_ms, std_ms) over BENCH_ITERS forward passes.

    All models run under bfloat16 autocast for a fair comparison.
    VideoMamba already forces bf16 internally; applying the context to others
    brings them to the same precision so numbers are directly comparable.
    """
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # Warmup
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(WARMUP_ITERS):
            model(**inputs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(BENCH_ITERS):
            start_evt.record()
            model(**inputs)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))  # ms

    return float(np.mean(times)), float(np.std(times))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARNING] No GPU found — latency numbers will be slow/unreliable.")

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"GPU: {gpu_name}")
    print(f"Benchmarking {len(MODEL_FRAMES)} models × {len(RESOLUTIONS)} resolutions "
          f"({WARMUP_ITERS} warmup + {BENCH_ITERS} bench iterations, batch=1)\n")

    rows = []
    for model_name in MODEL_FRAMES:
        for res in RESOLUTIONS:
            print(f"  {model_name:20s} @ {res:3d}px ... ", end="", flush=True)
            try:
                ckpt = ckpt_path(model_name, res)
                model, b = load(model_name, ckpt, device)
                inputs = make_input(b, model_name, res, device)
                mean_ms, std_ms = measure(model, inputs, device)
                print(f"{mean_ms:6.1f} ± {std_ms:.1f} ms  [{b}]")
                rows.append({
                    "model":      model_name,
                    "resolution": res,
                    "mean_ms":    round(mean_ms, 3),
                    "std_ms":     round(std_ms,  3),
                    "gpu":        gpu_name,
                    "backend":    b,
                    "n_frames":   MODEL_FRAMES[model_name],
                })
                # Free VRAM between runs
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"FAILED — {e}")
                rows.append({
                    "model": model_name, "resolution": res,
                    "mean_ms": None, "std_ms": None,
                    "gpu": gpu_name, "backend": "error", "n_frames": MODEL_FRAMES[model_name],
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
