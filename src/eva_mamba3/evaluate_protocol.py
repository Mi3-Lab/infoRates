"""Evaluate EVA models under the previous paper's coverage/stride protocol."""
from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .factory import BACKBONE_REGISTRY
from .protocol_data import build_protocol_loader

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFESTS = {
    "autsl": "evaluations/accv2026/manifests/autsl_val_20_per_class.csv",
    "finegym": "evaluations/accv2026/manifests/finegym_val_20_per_class.csv",
    "ssv2": "evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv",
    "ucf101": "evaluations/accv2026/manifests/ucf101_val_20_per_class.csv",
    "hmdb51": "evaluations/accv2026/manifests/hmdb51_val_20_per_class.csv",
    "diving48": "evaluations/accv2026/manifests/diving48_val_20_per_class.csv",
}


def configure_accelerator() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_checkpoint(model, checkpoint: str | None, device: torch.device) -> None:
    if checkpoint is None:
        return
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def eval_one(
    model,
    loader,
    device: torch.device,
    model_stride: int,
) -> tuple[float, int, float]:
    model.eval()
    correct = total = 0
    t0 = time.perf_counter()
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device):
            out = model(frames, s=model_stride)
        pred = out["logits"].argmax(-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    dt = time.perf_counter() - t0
    return (correct / total if total else 0.0), total, dt


def main() -> None:
    parser = argparse.ArgumentParser("Coverage/stride protocol evaluator for EVA")
    parser.add_argument("--dataset", default="finegym", choices=sorted(DEFAULT_MANIFESTS))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONE_REGISTRY))
    parser.add_argument("--model_name", default=None,
                        help="HF model id/path for TimeSformer-based backbones")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--mode", choices=["baseline", "eva_window"], required=True)
    parser.add_argument("--coverages", nargs="+", type=int, default=[100])
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    configure_accelerator()
    manifest = Path(args.manifest or DEFAULT_MANIFESTS[args.dataset])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_fn = BACKBONE_REGISTRY[args.backbone]
    if args.model_name:
        model = build_fn(n_classes=args.n_classes, model_name=args.model_name).to(device)
    else:
        model = build_fn(n_classes=args.n_classes).to(device)
    load_checkpoint(model, args.checkpoint, device)

    rows = []
    for coverage in args.coverages:
        for stride in args.strides:
            loader = build_protocol_loader(
                manifest=manifest,
                split=args.split,
                mode=args.mode,
                coverage=coverage,
                stride=stride,
                budget=args.budget,
                size=args.size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
            )
            model_stride = 1 if args.mode == "baseline" else stride
            acc, n, seconds = eval_one(model, loader, device, model_stride)
            row = {
                "dataset": args.dataset,
                "backbone": args.backbone,
                "mode": args.mode,
                "coverage": coverage,
                "stride": stride,
                "budget": args.budget,
                "top1": acc,
                "n": n,
                "seconds": seconds,
            }
            rows.append(row)
            print(
                f"cov={coverage:3d}% s={stride:2d} "
                f"top1={acc*100:.2f}% n={n} time={seconds:.1f}s",
                flush=True,
            )

    report = {
        "args": vars(args),
        "manifest": str(manifest),
        "rows": rows,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(out, "w") as f:
                json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
