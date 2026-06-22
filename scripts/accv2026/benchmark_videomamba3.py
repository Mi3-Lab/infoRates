#!/usr/bin/env python3
"""Benchmark VideoMamba3 variants on synthetic video tensors."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VM3 = ROOT / "experiments" / "videomamba3"
for path in (ROOT, ROOT / "src", VM3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from videomamba3 import VisionMamba, videomamba3_tiny, videomamba3_small, videomamba3_middle  # noqa: E402

EMBED_DIMS = {"tiny": 192, "small": 384, "middle": 576}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(args, variant: str) -> dict:
    factories = {
        "tiny": videomamba3_tiny,
        "small": videomamba3_small,
        "middle": videomamba3_middle,
    }
    device = torch.device(args.device)
    ssm_cfg = {
        "d_state": args.ssm_d_state,
        "expand": args.ssm_expand,
        "headdim": args.ssm_headdim,
        "mimo_rank": args.ssm_mimo_rank,
    }
    if args.depth > 0:
        model = VisionMamba(
            img_size=args.input_size,
            patch_size=16,
            depth=args.depth,
            embed_dim=EMBED_DIMS[args.model_size],
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            mamba3_variant=variant,
            mamba3_impl=args.mamba3_impl,
            ssm_cfg=ssm_cfg,
        ).to(device)
    else:
        model = factories[args.model_size](
            num_frames=args.num_frames,
            img_size=args.input_size,
            num_classes=args.num_classes,
            variant=variant,
            mamba3_impl=args.mamba3_impl,
            ssm_cfg=ssm_cfg,
        ).to(device)
    model.eval()
    if args.torch_compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    x = torch.randn(args.batch_size, 3, args.num_frames, args.input_size, args.input_size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x)
        synchronize(device)
        elapsed = time.perf_counter() - start

    params = sum(p.numel() for p in model.parameters())
    mean_s = elapsed / max(args.iters, 1)
    videos_s = args.batch_size / max(mean_s, 1e-9)
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    patches_per_frame = (args.input_size // 16) ** 2
    seq_len = 1 + args.num_frames * patches_per_frame
    return {
        "variant": variant,
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "input_size": args.input_size,
        "seq_len": seq_len,
        "depth": args.depth if args.depth > 0 else {"tiny": 24, "small": 24, "middle": 32}[args.model_size],
        "ssm_d_state": args.ssm_d_state,
        "ssm_expand": args.ssm_expand,
        "ssm_headdim": args.ssm_headdim,
        "ssm_mimo_rank": args.ssm_mimo_rank,
        "params": params,
        "mean_inference_s": mean_s,
        "videos_per_second": videos_s,
        "peak_memory_mb": peak_mem_mb,
        "device": str(device),
        "torch_compile": int(args.torch_compile),
        "mamba3_impl": args.mamba3_impl,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=["trapezoidal", "complex", "mimo"])
    p.add_argument("--mamba3-impl", default="auto", choices=["auto", "official", "reference"])
    p.add_argument("--model-size", default="tiny", choices=["tiny", "small", "middle"])
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=101)
    p.add_argument("--num-frames", type=int, default=2)
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--ssm-d-state", type=int, default=16)
    p.add_argument("--ssm-expand", type=int, default=1)
    p.add_argument("--ssm-headdim", type=int, default=32)
    p.add_argument("--ssm-mimo-rank", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--output", default="evaluations/accv2026/videomamba3/latency_benchmark.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = [benchmark_model(args, variant) for variant in args.variants]
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['variant']:12s} seq={row['seq_len']:4d} "
            f"mean={row['mean_inference_s']:.4f}s videos/s={row['videos_per_second']:.3f} "
            f"mem={row['peak_memory_mb']:.1f}MB params={row['params']:,}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
