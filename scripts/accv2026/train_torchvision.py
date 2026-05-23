#!/usr/bin/env python3
"""Fine-tune TorchVision 3D CNN video baselines on Something-Something V2."""

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

from info_rates.data.something import (  # noqa: E402
    get_class_mapping,
    get_numeric_labels,
    get_train_val_test_manifests,
    list_classes,
)
from info_rates.models.timesformer import UCFDataset  # noqa: E402
from info_rates.models.torchvision_video import (  # noqa: E402
    MODEL_REGISTRY,
    TorchvisionVideoProcessor,
    create_torchvision_video_model,
    save_torchvision_video_checkpoint,
)
from scripts.data_processing.train_multimodel import cleanup_ddp, setup_ddp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/Something_data")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="r3d_18")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=112)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-path", default="fine_tuned_models/accv2026_r3d18_ssv2")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="inforates-accv2026")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser.parse_args()


def prepare_data(data_root: str, max_train: int, max_val: int):
    labels_path = Path(data_root) / "labels" / "labels.json"
    class_names = list_classes(str(labels_path))
    train_df, val_df, _ = get_train_val_test_manifests(data_root)
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
    return class_names, train_files, val_files


def make_loader(files, processor, args, use_ddp: bool, train: bool):
    dataset = UCFDataset(files, processor, num_frames=args.num_frames, size=args.input_size)
    sampler = DistributedSampler(dataset, shuffle=train) if use_ddp else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def init_wandb(args, class_count: int, train_count: int, val_count: int) -> Optional[object]:
    if args.no_wandb or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb unavailable; continuing without W&B")
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"train-torchvision-{args.model}-{Path(args.save_path).name}",
        tags=args.wandb_tags,
        config={
            "dataset": "something-something-v2",
            "model": args.model,
            "architecture": "3d_cnn",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_frames": args.num_frames,
            "input_size": args.input_size,
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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int, show_progress: bool):
    model.train()
    total_loss = 0.0
    total_n = 0
    pbar = tqdm(loader, desc=f"epoch={epoch} train", disable=not show_progress)
    for batch in pbar:
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item()) * labels.numel()
        total_n += labels.numel()
        if show_progress:
            pbar.set_postfix(loss=total_loss / max(1, total_n))
    total_loss = reduce_sum(total_loss, device)
    total_n = reduce_sum(total_n, device)
    return total_loss / max(1.0, total_n)


@torch.inference_mode()
def evaluate(model, loader, device, show_progress: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_n = 0.0
    for batch in tqdm(loader, desc="validation", disable=not show_progress):
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
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

    class_names, train_files, val_files = prepare_data(args.data_root, args.max_train_samples, args.max_val_samples)
    if is_main:
        print(f"Classes: {len(class_names)} | Train: {len(train_files)} | Val: {len(val_files)}")

    processor = TorchvisionVideoProcessor(size=args.input_size)
    train_loader = make_loader(train_files, processor, args, args.ddp, train=True)
    val_loader = make_loader(val_files, processor, args, args.ddp, train=False)

    model = create_torchvision_video_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    wandb_run = init_wandb(args, len(class_names), len(train_files), len(val_files))

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        if args.ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, is_main)
        val_loss, val_acc = evaluate(model, val_loader, device, is_main)
        if is_main:
            print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if wandb_run is not None:
                import wandb

                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        best_acc = max(best_acc, val_acc)

    if is_main:
        raw_model = model.module if hasattr(model, "module") else model
        save_torchvision_video_checkpoint(
            args.save_path,
            raw_model,
            model_name=args.model,
            class_names=class_names,
            num_frames=args.num_frames,
            input_size=args.input_size,
        )
        print(f"Saved TorchVision video checkpoint to {args.save_path}; best_val_acc={best_acc:.4f}")
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
