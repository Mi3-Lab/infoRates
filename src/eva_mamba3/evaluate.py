"""Evaluation suite for EVA-Mamba3.

Computes:
  - TDS (Temporal Demand Score): acc[s=1] - acc[s=16]
  - AAG (Anti-Aliasing Gain): mean accuracy gain over baseline across sparse strides
  - Phase robustness: variance of predictions across n_phases temporal offsets
  - Pareto data: (latency_ms, accuracy) per (method, stride)
"""
from __future__ import annotations

import time
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

STRIDES = [1, 2, 4, 8, 16]
SPARSE_STRIDES = [2, 4, 8, 16]


def configure_accelerator() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _amp(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ── Per-stride accuracy ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(
    model,
    loader,
    device: torch.device,
    strides: List[int] = STRIDES,
    max_batches: int | None = None,
) -> Dict[int, float]:
    """Returns {stride: top1_accuracy} for each stride in strides."""
    model.eval()
    results: Dict[int, float] = {}
    for s in strides:
        correct = total = 0
        for batch_idx, (frames, labels) in enumerate(loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _amp(device):
                out = model(frames, s=s)
            pred = out["logits"].argmax(-1)
            correct += (pred == labels).sum().item()
            total   += len(labels)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
        results[s] = correct / total if total > 0 else 0.0
    return results


# ── TDS ───────────────────────────────────────────────────────────────────────

def tds(acc: Dict[int, float]) -> float:
    """Temporal Demand Score = max(0, acc[s=1] - acc[s=16])."""
    return max(0.0, acc.get(1, 0.0) - acc.get(16, 0.0))


# ── AAG ───────────────────────────────────────────────────────────────────────

def aag(
    acc_model:    Dict[int, float],
    acc_baseline: Dict[int, float],
    strides: List[int] = SPARSE_STRIDES,
) -> float:
    """Anti-Aliasing Gain: mean acc improvement over baseline at sparse strides."""
    gains = [acc_model.get(s, 0.0) - acc_baseline.get(s, 0.0) for s in strides]
    return sum(gains) / len(gains)


# ── Phase robustness ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_phase_robustness(
    model,
    loader,
    device:   torch.device,
    s:        int = 4,
    n_phases: int = 8,
    max_batches: int | None = None,
) -> float:
    """Mean variance of softmax predictions across n_phases temporal offsets.

    Lower = more robust to temporal phase of the stride grid.
    """
    model.eval()
    variances: List[float] = []

    for batch_idx, (frames, _) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        B, T = frames.shape[:2]

        probs_list: List[Tensor] = []
        for phi in range(n_phases):
            offset = int(phi * s / n_phases)
            f_shifted = torch.roll(frames, shifts=-offset, dims=1)
            with _amp(device):
                out = model(f_shifted, s=s)
            probs_list.append(out["logits"].float().softmax(-1))   # (B, C)

        probs = torch.stack(probs_list, dim=0)   # (n_phases, B, C)
        var   = probs.var(dim=0).mean().item()   # mean variance over classes/batch
        variances.append(var)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return sum(variances) / len(variances) if variances else 0.0


# ── Latency measurement ───────────────────────────────────────────────────────

@torch.no_grad()
def measure_latency(
    model,
    device:   torch.device,
    batch_size: int = 1,
    T:        int = 64,
    H:        int = 224,
    W:        int = 224,
    s:        int = 4,
    n_warmup: int = 20,
    n_bench:  int = 100,
) -> float:
    """Returns mean end-to-end model latency in milliseconds."""
    model.eval()
    dummy = torch.randint(0, 255, (batch_size, T, H, W, 3), dtype=torch.uint8, device=device)

    # Warmup
    for _ in range(n_warmup):
        with _amp(device):
            _ = model(dummy, s=s)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n_bench):
        with _amp(device):
            _ = model(dummy, s=s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    return (t_end - t_start) / n_bench * 1000.0   # ms


# ── Full evaluation report ────────────────────────────────────────────────────

def full_report(
    model,
    loader,
    device:           torch.device,
    baseline_acc:     Dict[int, float] | None = None,
    measure_lat:      bool = True,
    max_batches:      int | None = None,
) -> dict:
    """Run all metrics and return a summary dict."""
    acc = eval_accuracy(model, loader, device, max_batches=max_batches)
    report = {
        "accuracy": acc,
        "tds":      tds(acc),
    }

    if baseline_acc is not None:
        report["aag"] = aag(acc, baseline_acc)

    report["phase_var_s4"] = eval_phase_robustness(
        model, loader, device, s=4, max_batches=max_batches
    )

    if measure_lat:
        lats = {s: measure_latency(model, device, s=s) for s in STRIDES}
        report["latency_ms"] = lats
        # Pareto: list of (latency, accuracy) pairs
        report["pareto"] = [(lats[s], acc[s]) for s in STRIDES]

    return report


def _load_checkpoint(model, checkpoint: str | None, device: torch.device) -> None:
    if checkpoint is None:
        return
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate EVA models")
    parser.add_argument("--dataset", default="finegym")
    parser.add_argument("--n_classes", type=int, default=97)
    parser.add_argument("--backbone", default="mamba3")
    parser.add_argument("--model_name", default=None,
                        help="HF model id/path for TimeSformer-based backbones")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--baseline_json", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--no_latency", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    configure_accelerator()
    from .datasets import build_dataloader
    from .factory import BACKBONE_REGISTRY

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_fn = BACKBONE_REGISTRY[args.backbone]
    if args.model_name:
        model = build_fn(n_classes=args.n_classes, model_name=args.model_name).to(device)
    else:
        model = build_fn(n_classes=args.n_classes).to(device)
    _load_checkpoint(model, args.checkpoint, device)

    loader = build_dataloader(
        args.dataset,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        T=args.T,
        data_root=args.data_root,
    )

    baseline_acc = None
    if args.baseline_json is not None:
        with open(args.baseline_json) as f:
            raw = json.load(f)
        source = raw.get("accuracy", raw)
        baseline_acc = {int(k): float(v) for k, v in source.items()}

    report = full_report(
        model,
        loader,
        device,
        baseline_acc=baseline_acc,
        measure_lat=not args.no_latency,
        max_batches=args.max_batches,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
