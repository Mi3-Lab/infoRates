#!/usr/bin/env python3
"""Fine-tune SlowFast R50 on Something-Something V2 for ACCV 2026."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
DATA_PROCESSING = SCRIPTS / "data_processing"
for path in (ROOT, SRC, SCRIPTS, DATA_PROCESSING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from info_rates.data.something import (
    get_class_mapping,
    get_numeric_labels,
    get_train_val_test_manifests,
    list_classes,
)
from info_rates.models.timesformer import UCFDataset
from info_rates.models.slowfast_video import (
    SlowFastVideoProcessor,
    create_slowfast_model,
    load_slowfast_checkpoint,
    save_slowfast_checkpoint,
    SLOWFAST_FAST_FRAMES,
)
from info_rates.training.ddp import cleanup_ddp, setup_ddp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ssv2",
                        choices=["ssv2", "ucf101", "hmdb51", "diving48", "wlasl", "wlasl100", "epic_kitchens", "autsl", "driveact", "flame", "ufc_crime", "finegym", "ego4d"],
                        help="Dataset to train on (default: ssv2)")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (auto-detected from --dataset if omitted)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-path", default=None,
                        help="Checkpoint dir (auto-generated as accv2026_slowfast_r50_<dataset> if omitted)")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume-from", default=None, help="Checkpoint dir to resume from (epoch parsed from dir name _epochN)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="inforates-accv2026")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args()


_DEFAULT_DATA_ROOTS = {
    "ssv2": "data/Something_data",
    "ucf101": "data/UCF101_data",
    "hmdb51": "data/HMDB51_data",
    "diving48": "data/Diving48_data",
    "wlasl": "data/WLASL_data",
    "wlasl100": "data/WLASL_data",
    "epic_kitchens": "data/EPIC_data",
    "autsl": "data/AUTSL_data",
    "driveact": "data/DriveAct_data",
    "flame": "data/FLAME_data",
    "ufc_crime": "data/UCFCrime_data",
    "finegym":    "data/FineGym_data",
    "ego4d":      "data/Ego4D_data",
}


def prepare_data(dataset: str, data_root: str | None, max_train: int, max_val: int):
    root = data_root or _DEFAULT_DATA_ROOTS.get(dataset, "data/Something_data")

    if dataset == "ssv2":
        labels_path = Path(root) / "labels" / "labels.json"
        class_names = list_classes(str(labels_path))
        train_df, val_df, _ = get_train_val_test_manifests(root)
        mapping = get_class_mapping(str(labels_path))
        train_df = get_numeric_labels(train_df, mapping)
        val_df = get_numeric_labels(val_df, mapping)
        train_df = train_df[train_df["video_path"].apply(os.path.exists)].copy()
        val_df = val_df[val_df["video_path"].apply(os.path.exists)].copy()
        if max_train > 0:
            train_df = train_df.iloc[:max_train].copy()
        if max_val > 0:
            val_df = val_df.iloc[:max_val].copy()
        train_files = list(zip(train_df["video_path"].tolist(), train_df["label"].astype(int).tolist()))
        val_files = list(zip(val_df["video_path"].tolist(), val_df["label"].astype(int).tolist()))
    else:
        from info_rates.data.datasets import load_dataset
        class_names, train_files, val_files = load_dataset(dataset, root)
        train_files = [(p, l) for p, l in train_files if os.path.exists(p)]
        val_files = [(p, l) for p, l in val_files if os.path.exists(p)]
        if max_train > 0:
            train_files = train_files[:max_train]
        if max_val > 0:
            val_files = val_files[:max_val]

    return class_names, train_files, val_files


def make_loader(files, processor, args, use_ddp: bool, train: bool):
    # UCFDataset decodes `num_frames` from the video; SlowFast adapts internally
    dataset = UCFDataset(files, processor, num_frames=SLOWFAST_FAST_FRAMES, size=args.input_size)
    sampler = DistributedSampler(dataset, shuffle=train) if use_ddp else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,  # Disabled due to ulimit -l 8MB (max locked memory)
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )


def init_wandb(args, class_count: int, train_count: int, val_count: int) -> Optional[object]:
    if args.no_wandb or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb unavailable")
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"train-slowfast-r50-{Path(args.save_path).name}",
        tags=args.wandb_tags,
        config={
            "dataset": args.dataset,
            "model": "slowfast_r50",
            "architecture": "slowfast",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "input_size": args.input_size,
            "slow_frames": 8,
            "fast_frames": SLOWFAST_FAST_FRAMES,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "class_count": class_count,
            "train_count": train_count,
            "val_count": val_count,
            "phase": "train",
            "ddp": args.ddp,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("ACCV_JOB_ID"),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
    )


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def train_one_epoch(model, loader, optimizer, device, epoch, show_progress):
    model.train()
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc=f"epoch={epoch} train", disable=not show_progress)
    for batch in pbar:
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * labels.numel()
        total_n += labels.numel()
        if show_progress:
            pbar.set_postfix(loss=total_loss / max(1, total_n))
    total_loss = reduce_sum(total_loss, device)
    total_n = reduce_sum(total_n, device)
    return total_loss / max(1.0, total_n)


@torch.inference_mode()
def evaluate(model, loader, device, show_progress):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0.0, 0.0
    for batch in tqdm(loader, desc="validation", disable=not show_progress):
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        total_correct += float((logits.argmax(dim=-1) == labels).sum().item())
        total_n += float(labels.numel())
    total_loss = reduce_sum(total_loss, device)
    total_correct = reduce_sum(total_correct, device)
    total_n = reduce_sum(total_n, device)
    return total_loss / max(1.0, total_n), total_correct / max(1.0, total_n)


def main() -> None:
    args = parse_args()
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if args.save_path is None:
        args.save_path = f"fine_tuned_models/accv2026_slowfast_r50_{args.dataset}"

    class_names, train_files, val_files = prepare_data(args.dataset, args.data_root, args.max_train_samples, args.max_val_samples)
    if is_main:
        print(f"Classes: {len(class_names)} | Train: {len(train_files)} | Val: {len(val_files)}")

    processor = SlowFastVideoProcessor(size=args.input_size)
    train_loader = make_loader(train_files, processor, args, args.ddp, train=True)
    val_loader = make_loader(val_files, processor, args, args.ddp, train=False)

    start_epoch = 1
    model = create_slowfast_model(len(class_names), pretrained=not args.no_pretrained).to(device)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if (resume_path / "config.json").exists():
            model, _, _ = load_slowfast_checkpoint(str(resume_path), device=str(device))
            model.to(device)
            # Parse epoch from checkpoint name (_epochN suffix)
            import re
            m = re.search(r"_epoch(\d+)$", resume_path.name)
            start_epoch = int(m.group(1)) + 1 if m else 2
            if is_main:
                print(f"[Resume] Loaded from {resume_path}, starting at epoch {start_epoch}")
        else:
            if is_main:
                print(f"[Resume] Checkpoint not found at {resume_path}, starting from scratch")

    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # SlowFast lateral connections cause the same weight to accumulate
        # gradients twice per backward; _set_static_graph() handles this correctly.
        model._set_static_graph()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb_run = init_wandb(args, len(class_names), len(train_files), len(val_files))

    best_acc = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        if args.ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, is_main)
        val_loss, val_acc = evaluate(model, val_loader, device, is_main)
        if is_main:
            print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if wandb_run is not None:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
            raw_model = model.module if hasattr(model, "module") else model
            epoch_ckpt = f"{args.save_path}_epoch{epoch}"
            save_slowfast_checkpoint(
                epoch_ckpt, raw_model,
                class_names=class_names,
                num_frames=SLOWFAST_FAST_FRAMES,
                input_size=args.input_size,
            )
            print(f"  -> Saved epoch checkpoint: {epoch_ckpt}")
        best_acc = max(best_acc, val_acc)

    if is_main:
        raw_model = model.module if hasattr(model, "module") else model
        save_slowfast_checkpoint(
            args.save_path,
            raw_model,
            class_names=class_names,
            num_frames=SLOWFAST_FAST_FRAMES,
            input_size=args.input_size,
        )
        print(f"Saved SlowFast checkpoint to {args.save_path}; best_val_acc={best_acc:.4f}")
        if wandb_run is not None:
            import wandb
            wandb.summary["checkpoint_path"] = args.save_path
            wandb.summary["best_val_accuracy"] = best_acc
            wandb.save(str(Path(args.save_path) / "config.json"), policy="now")
            wandb_run.finish()

    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
