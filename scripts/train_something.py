#!/usr/bin/env python3
"""
Minimal, robust fine-tuning script for Something-Something V2.

Usage examples:
  # Single-GPU
  python scripts/train_something.py --model timesformer --epochs 5 --batch-size 8 --lr 2e-5 --save-path fine_tuned_models/something

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node=2 scripts/train_something.py --model timesformer --epochs 5 --batch-size 4 --ddp

Notes:
 - Script filters out missing videos automatically.
 - Corrupted videos that raise errors are removed during dataset access (UCFDataset behavior).
 - Defaults are chosen to be conservative; tune hyperparams for your machine.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Ensure project, scripts and src on path (like train_multimodel)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
DATA_PROCESSING = SCRIPTS / "data_processing"
for p in (str(ROOT), str(SRC), str(SCRIPTS), str(DATA_PROCESSING)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from info_rates.data.something import (
    list_classes,
    get_train_val_test_manifests,
    get_class_mapping,
    get_numeric_labels,
)
from info_rates.models.timesformer import build_dataloaders, save_model
from model_factory import ModelFactory
from scripts.data_processing.train_multimodel import fine_tune_model, setup_ddp, cleanup_ddp


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Time/Video models on Something-Something V2")
    p.add_argument("--data-root", default="data/Something_data", help="Root of Something-Something data")
    p.add_argument("--model", choices=["timesformer", "videomae", "vivit"], default="timesformer")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--save-path", type=str, default="fine_tuned_models/something")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ddp", action="store_true", help="Use DDP (launch with torchrun)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--cleanup-interval", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--use-dummy-model", action="store_true", help="Use tiny dummy model for smoke tests")
    p.add_argument("--smoke-test", action="store_true", help="Run a smoke test without real videos (small dummy dataset)")
    return p.parse_args()


def prepare_data(data_root: str, max_train: int = None, max_val: int = None):
    labels_path = os.path.join(data_root, "labels/labels.json")
    class_names = list_classes(labels_path)
    train_df, val_df, _ = get_train_val_test_manifests(data_root)
    mapping = get_class_mapping(labels_path)
    train_df = get_numeric_labels(train_df, mapping)
    val_df = get_numeric_labels(val_df, mapping)

    # Filter missing files
    train_df = train_df[train_df["video_path"].apply(os.path.exists)].copy()
    val_df = val_df[val_df["video_path"].apply(os.path.exists)].copy()

    if max_train:
        train_df = train_df.iloc[:max_train]
    if max_val:
        val_df = val_df.iloc[:max_val]

    train_files = list(zip(train_df["video_path"].tolist(), train_df["label"].tolist()))
    val_files = list(zip(val_df["video_path"].tolist(), val_df["label"].tolist()))

    return class_names, train_files, val_files


def main():
    args = parse_args()

    # Force using CUDA when requested
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Initialize DDP if requested (when using torchrun)
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        # set device to per-process CUDA device
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Smoke test path: small dummy dataset + dummy model to validate loop
    if args.smoke_test:
        print("[INFO] Running smoke test (dummy data + dummy model)")
        # Create tiny dummy datasets
        from torch.utils.data import Dataset, DataLoader
        class DummyDataset(Dataset):
            def __init__(self, n, num_frames=4, H=16, W=16):
                self.n = n
                self.num_frames = num_frames
                self.H = H
                self.W = W
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                # Return frames shaped (T, H, W, C) matching many video processors
                pv = torch.randint(0, 255, (self.num_frames, self.H, self.W, 3), dtype=torch.uint8).float() / 255.0
                label = torch.tensor(idx % 10, dtype=torch.long)
                return {"pixel_values": pv, "labels": label}

        # Load model info and processor early so DummyDataset can use the processor
        model_info = ModelFactory.get_model_info(args.model)
        num_frames = model_info["default_frames"]
        processor = ModelFactory.load_processor(args.model)

        class_names = [str(i) for i in range(10)]
        class DummyDataset(Dataset):
            def __init__(self, n, num_frames=4, H=16, W=16, processor=None):
                self.n = n
                self.num_frames = num_frames
                self.H = H
                self.W = W
                self.processor = processor
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                # Create random frames as uint8 images and let the processor convert them
                import numpy as np
                frames = [np.random.randint(0, 255, (self.H, self.W, 3), dtype=np.uint8) for _ in range(self.num_frames)]
                inputs = self.processor(frames, return_tensors="pt")
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor(idx % 10, dtype=torch.long)
                return inputs

        train_dl = DataLoader(DummyDataset(8, num_frames=min(4, num_frames), processor=processor), batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(DummyDataset(4, num_frames=min(4, num_frames), processor=processor), batch_size=args.batch_size, shuffle=False)

    else:
        print(f"[INFO] Preparing Something-Something V2 data from {args.data_root}")
        class_names, train_files, val_files = prepare_data(args.data_root, args.max_train_samples, args.max_val_samples)

        if len(train_files) == 0 or len(val_files) == 0:
            print("[ERROR] No valid train/val files found. Check data paths and that videos are present.")
            return

        print(f"[INFO] Classes: {len(class_names)} | Train samples: {len(train_files)} | Val samples: {len(val_files)}")

        # Get model info and processor
        model_info = ModelFactory.get_model_info(args.model)
        num_frames = model_info["default_frames"]
        processor = ModelFactory.load_processor(args.model)

        # Build dataloaders
        train_dl, val_dl = build_dataloaders(
            train_files=train_files,
            val_files=val_files,
            class_names=class_names,
            processor=processor,
            batch_size=args.batch_size,
            num_frames=num_frames,
            size=224,
            use_ddp=args.ddp,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Fine-tune
    model = fine_tune_model(
        train_dl=train_dl,
        val_dl=val_dl,
        num_classes=len(class_names),
        model_name=args.model,
        model_id=model_info["model_id"],
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        use_wandb=(not args.no_wandb),
        use_ddp=args.ddp,
        local_rank=local_rank,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cleanup_interval=args.cleanup_interval,
        use_dummy_model=False,
    )

    # Save model (only on main process in DDP)
    if not args.ddp or (args.ddp and int(os.environ.get('LOCAL_RANK', '0')) == 0):
        os.makedirs(args.save_path, exist_ok=True)
        try:
            save_model(args.save_path, model.module if hasattr(model, "module") else model, processor, class_names)
            print(f"[INFO] Saved fine-tuned model to: {args.save_path}")
        except Exception as e:
            print(f"[WARN] Failed to call save_model: {e}")

    # Cleanup DDP if initialized
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
