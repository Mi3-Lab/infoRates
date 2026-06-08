#!/usr/bin/env python3
"""Orchestrate training of 8 models across datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/accv2026"


# SAFE batch sizes for RTX 6000 Pro (102GB VRAM)
MODELS = [
    ("torchvision", "r3d_18", dict(frames=16, batch_size=64)),
    ("torchvision", "mc3_18", dict(frames=16, batch_size=64)),
    ("torchvision", "r2plus1d_18", dict(frames=16, batch_size=64)),
    ("slowfast", "slowfast_r50", dict(frames=32, batch_size=32)),
    ("transformers", "timesformer", dict(frames=8, batch_size=64)),
    ("transformers", "vivit", dict(frames=32, batch_size=32)),
    ("transformers", "videomae", dict(frames=16, batch_size=64)),
    ("videomamba", "videomamba", dict(frames=8, batch_size=8)),
]


def run_cmd(cmd: list[str], dry_run: bool = False) -> int:
    """Run command and return exit code."""
    print(f"+ {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd)


def train_dataset(dataset: str, epochs: int = 10, dry_run: bool = False, skip_models: list = None):
    """Train all 8 models on a dataset."""
    skip_models = skip_models or []

    print(f"\n{'='*80}")
    print(f"Training all 8 models on {dataset.upper()} (epochs={epochs})")
    print(f"{'='*80}\n")

    for backend, model_name, cfg in MODELS:
        if model_name in skip_models:
            print(f"[SKIP] {model_name}")
            continue

        if backend == "torchvision":
            cmd = [
                sys.executable,
                str(SCRIPTS / "train_torchvision.py"),
                "--dataset", dataset,
                "--model", model_name,
                "--epochs", str(epochs),
                "--batch-size", str(cfg["batch_size"]),
                "--num-frames", str(cfg["frames"]),
                "--num-workers", "4",
            ]
        elif backend == "slowfast":
            cmd = [
                sys.executable,
                str(SCRIPTS / "train_slowfast.py"),
                "--dataset", dataset,
                "--epochs", str(epochs),
                "--batch-size", str(cfg["batch_size"]),
                "--num-workers", "4",
            ]
        elif backend == "transformers":
            # Transformers use model-default frames, no --num-frames arg
            cmd = [
                sys.executable,
                str(SCRIPTS / "train_transformers.py"),
                "--dataset", dataset,
                "--model", model_name,
                "--epochs", str(epochs),
                "--batch-size", str(cfg["batch_size"]),
                "--num-workers", "4",
            ]
        elif backend == "videomamba":
            cmd = [
                sys.executable,
                str(SCRIPTS / "train_videomamba.py"),
                "--dataset", dataset,
                "--epochs", str(epochs),
                "--batch-size", str(cfg["batch_size"]),
            ]
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if "--wandb-tags" not in cmd and backend != "videomamba":
            cmd.extend(["--wandb-tags", "accv2026", "dataset_expansion", dataset])

        print(f"[{backend:15s}] Training {model_name} (batch_size={cfg['batch_size']})...")
        ret = run_cmd(cmd, dry_run=dry_run)
        if ret != 0:
            print(f"  ❌ {model_name} FAILED (exit code {ret})")
        else:
            print(f"  ✓ {model_name} DONE")
        print()

    print(f"{'='*80}")
    print(f"Finished training all 8 models on {dataset.upper()}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["flame", "ufc_crime", "finegym"],
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-models", nargs="+", default=[],
                        help="Models to skip (e.g., --skip-models videomamba timesformer)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_dataset(args.dataset, epochs=args.epochs, dry_run=args.dry_run, skip_models=args.skip_models)


if __name__ == "__main__":
    main()
