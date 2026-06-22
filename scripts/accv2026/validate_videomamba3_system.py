#!/usr/bin/env python3
"""System-level VideoMamba3 validation: latency, throughput, train-step VRAM.

This is intentionally synthetic and controlled. Accuracy comes from dataset
training/eval; this script answers whether a configuration is fast enough to
deserve larger experiments.
"""
from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
VM3 = ROOT / "experiments" / "videomamba3"
for path in (ROOT, ROOT / "src", VM3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from videomamba3 import VisionMamba  # noqa: E402


EMBED_DIMS = {"tiny": 192, "small": 384, "middle": 576}
DEFAULT_DEPTHS = {"tiny": 24, "small": 24, "middle": 32}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / (1024**2),
        torch.cuda.max_memory_reserved(device) / (1024**2),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def make_model(args, variant: str, depth: int) -> torch.nn.Module:
    ssm_cfg = {
        "d_state": args.ssm_d_state,
        "expand": args.ssm_expand,
        "headdim": args.ssm_headdim,
        "mimo_rank": args.ssm_mimo_rank,
    }
    return VisionMamba(
        img_size=args.input_size,
        patch_size=16,
        depth=depth,
        embed_dim=EMBED_DIMS[args.model_size],
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        mamba3_variant=variant,
        mamba3_impl=args.mamba3_impl,
        ssm_cfg=ssm_cfg,
    )


def run_case(args, variant: str, depth: int, mode: str) -> dict:
    device = torch.device(args.device)
    row = {
        "status": "ok",
        "error": "",
        "variant": variant,
        "model_size": args.model_size,
        "depth": depth,
        "mode": mode,
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "input_size": args.input_size,
        "seq_len": 1 + args.num_frames * (args.input_size // 16) ** 2,
        "num_classes": args.num_classes,
        "ssm_d_state": args.ssm_d_state,
        "ssm_expand": args.ssm_expand,
        "ssm_headdim": args.ssm_headdim,
        "ssm_mimo_rank": args.ssm_mimo_rank,
        "amp": int(args.amp),
        "torch_compile": int(args.torch_compile),
        "mamba3_impl": args.mamba3_impl,
        "params": 0,
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_s": 0.0,
        "median_s": 0.0,
        "p95_s": 0.0,
        "videos_per_second": 0.0,
        "tokens_per_second": 0.0,
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        model = make_model(args, variant, depth).to(device)
        row["params"] = sum(p.numel() for p in model.parameters())
        if args.torch_compile:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        x = torch.randn(args.batch_size, 3, args.num_frames, args.input_size, args.input_size, device=device)
        labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05) if mode == "train" else None

        def step() -> torch.Tensor:
            enabled = args.amp and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
                logits = model(x)
                if mode == "train":
                    return F.cross_entropy(logits, labels)
                return logits

        if mode == "train":
            model.train()
            for _ in range(args.warmup):
                optimizer.zero_grad(set_to_none=True)
                loss = step()
                loss.backward()
                optimizer.step()
            sync(device)
            times = []
            for _ in range(args.iters):
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = step()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sync(device)
                times.append(time.perf_counter() - start)
        else:
            model.eval()
            with torch.inference_mode():
                for _ in range(args.warmup):
                    _ = step()
                sync(device)
                times = []
                for _ in range(args.iters):
                    start = time.perf_counter()
                    _ = step()
                    sync(device)
                    times.append(time.perf_counter() - start)

        row["mean_s"] = statistics.mean(times)
        row["median_s"] = statistics.median(times)
        row["p95_s"] = percentile(times, 0.95)
        row["videos_per_second"] = args.batch_size / max(row["mean_s"], 1e-9)
        row["tokens_per_second"] = args.batch_size * row["seq_len"] / max(row["mean_s"], 1e-9)
        row["peak_allocated_mb"], row["peak_reserved_mb"] = memory_mb(device)
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "oom"
        row["error"] = str(exc).splitlines()[0]
        if device.type == "cuda":
            row["peak_allocated_mb"], row["peak_reserved_mb"] = memory_mb(device)
            torch.cuda.empty_cache()
    except Exception as exc:  # Keep the sweep alive and inspect failures in CSV.
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=["complex"])
    p.add_argument("--mamba3-impl", default="auto", choices=["auto", "official", "reference"])
    p.add_argument("--model-size", default="tiny", choices=["tiny", "small", "middle"])
    p.add_argument("--depths", nargs="+", type=int, default=[4, 8, 12])
    p.add_argument("--modes", nargs="+", default=["inference", "train"], choices=["inference", "train"])
    p.add_argument("--num-classes", type=int, default=101)
    p.add_argument("--num-frames", type=int, default=2)
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--ssm-d-state", type=int, default=16)
    p.add_argument("--ssm-expand", type=int, default=1)
    p.add_argument("--ssm-headdim", type=int, default=32)
    p.add_argument("--ssm-mimo-rank", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="evaluations/accv2026/videomamba3/system_validation.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for variant in args.variants:
        for depth in args.depths:
            depth = DEFAULT_DEPTHS[args.model_size] if depth <= 0 else depth
            for mode in args.modes:
                row = run_case(args, variant, depth, mode)
                rows.append(row)
                print(
                    f"{row['status']:>5s} {variant:12s} depth={depth:<2d} {mode:9s} "
                    f"mean={row['mean_s']:.4f}s videos/s={row['videos_per_second']:.3f} "
                    f"alloc={row['peak_allocated_mb']:.1f}MB reserved={row['peak_reserved_mb']:.1f}MB"
                )
                if row["error"]:
                    print(f"      {row['error']}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
