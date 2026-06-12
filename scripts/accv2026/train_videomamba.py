#!/usr/bin/env python3
"""Fine-tune VideoMamba (Mamba SSM) on video classification datasets — ACCV 2026."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VIDEOMAMBA_REPO = ROOT / "third_party" / "videomamba_repo" / "videomamba" / "video_sm"
MAMBA_BUNDLED = ROOT / "third_party" / "videomamba_repo" / "mamba"
for _p in (ROOT, SRC, VIDEOMAMBA_REPO, MAMBA_BUNDLED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2
import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from info_rates.data.something import (
    get_class_mapping,
    get_numeric_labels,
    get_train_val_test_manifests,
    list_classes,
)
from info_rates.models.videomamba_model import (
    PRETRAINED_PATH,
    VideoMambaProcessor,
    build_videomamba,
    build_videomamba_from_finetuned,
    load_videomamba_checkpoint,
)
from info_rates.training.ddp import cleanup_ddp, setup_ddp

BAD_VIDEOS_LOG = ROOT / "evaluations/accv2026/logs/bad_videos.tsv"

_DEFAULT_DATA_ROOTS = {
    "ssv2":          "data/Something_data",
    "ucf101":        "data/UCF101_data",
    "hmdb51":        "data/HMDB51_data",
    "diving48":      "data/Diving48_data",
    "epic_kitchens": "data/EPIC_data",
    "autsl":         "data/AUTSL_data",
    "driveact":      "data/DriveAct_data",
    "flame":         "data/FLAME_data",
    "ufc_crime":     "data/UCFCrime_data",
    "finegym":       "data/FineGym_data",
    "ego4d":         "data/Ego4D_data",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoMambaDataset(Dataset):
    """Decodes video frames and returns tensors ready for VideoMamba."""

    def __init__(
        self,
        files: list,
        processor: VideoMambaProcessor,
        num_frames: int = 8,
        max_decode_retries: int | None = None,
    ):
        self.files = files
        self.processor = processor
        self.num_frames = num_frames
        self.max_decode_retries = max_decode_retries or int(
            os.environ.get("INFORATES_MAX_DECODE_RETRIES", "8")
        )
        self.bad_video_log = os.environ.get(
            "INFORATES_BAD_VIDEO_LOG",
            str(BAD_VIDEOS_LOG),
        )
        self._reported_bad: set = set()
        # files is list of (path, label) tuples
        self.has_labels = isinstance(self.files[0], tuple) if self.files else False

    def __len__(self):
        return len(self.files)

    def _resolve(self, idx):
        if self.has_labels:
            return self.files[idx]
        path = self.files[idx]
        label_name = os.path.basename(os.path.dirname(path))
        return path, 0

    def _log_bad(self, path, error):
        if path not in self._reported_bad:
            print(f"[WARN] Skipping bad video: {path} ({error})", flush=True)
            self._reported_bad.add(path)
        try:
            os.makedirs(os.path.dirname(self.bad_video_log), exist_ok=True)
            with open(self.bad_video_log, "a") as f:
                f.write(f"{path}\t{type(error).__name__}\t{str(error).splitlines()[0]}\n")
        except Exception:
            pass

    def _decode(self, path) -> np.ndarray:
        from decord import VideoReader, cpu
        # num_threads=1 prevents VP8/VP9 webm EAGAIN errors from FFmpeg thread pool.
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total = len(vr)
        if total <= 0:
            raise RuntimeError(f"Video has 0 frames: {path}")
        idxs = np.linspace(0, total - 1, self.num_frames).astype(int)
        return vr.get_batch(idxs).asnumpy()  # (T, H, W, C)

    def __getitem__(self, idx):
        attempts = min(max(1, self.max_decode_retries), max(1, len(self.files)))
        last_error = None
        for i in range(attempts):
            # First attempt uses the requested index; retries jump to random indices
            # to avoid sequential clusters of corrupt videos (common in SSv2 webm).
            try_idx = idx if i == 0 else random.randint(0, len(self.files) - 1)
            path, label = self._resolve(try_idx)
            try:
                frames = self._decode(path)
                break
            except Exception as e:
                last_error = e
                self._log_bad(path, e)
        else:
            raise RuntimeError(f"Failed after {attempts} attempts: {last_error}")

        inputs = self.processor([frames[i] for i in range(len(frames))], return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, T, H, W)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bad_video_paths() -> set:
    if not BAD_VIDEOS_LOG.exists():
        return set()
    bad = set()
    with open(BAD_VIDEOS_LOG) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                bad.add(parts[0])
    return bad


def prepare_data(dataset: str, data_root: str | None, max_train: int, max_val: int):
    root = data_root or _DEFAULT_DATA_ROOTS.get(dataset, "data/Something_data")

    if dataset == "ssv2":
        labels_path = Path(root) / "labels" / "labels.json"
        class_names = list_classes(str(labels_path))
        train_df, val_df, _ = get_train_val_test_manifests(root)
        mapping = get_class_mapping(str(labels_path))
        train_df = get_numeric_labels(train_df, mapping)
        val_df   = get_numeric_labels(val_df, mapping)
        train_df = train_df[train_df["video_path"].apply(os.path.exists)].copy()
        val_df   = val_df[val_df["video_path"].apply(os.path.exists)].copy()
        if max_train > 0:
            train_df = train_df.iloc[:max_train].copy()
        if max_val > 0:
            val_df = val_df.iloc[:max_val].copy()
        train_files = list(zip(train_df["video_path"].tolist(), train_df["label"].astype(int).tolist()))
        val_files   = list(zip(val_df["video_path"].tolist(),   val_df["label"].astype(int).tolist()))
    else:
        from info_rates.data.datasets import load_dataset
        class_names, train_files, val_files = load_dataset(dataset, root)
        train_files = [(p, l) for p, l in train_files if os.path.exists(p)]
        val_files   = [(p, l) for p, l in val_files   if os.path.exists(p)]
        if max_train > 0:
            train_files = train_files[:max_train]
        if max_val > 0:
            val_files = val_files[:max_val]

    bad_videos = load_bad_video_paths()
    if bad_videos:
        before = len(train_files)
        train_files = [(p, l) for p, l in train_files if p not in bad_videos]
        val_files   = [(p, l) for p, l in val_files   if p not in bad_videos]
        removed = before - len(train_files)
        if removed:
            print(f"[DataFilter] Excluded {removed} known-bad videos")

    return class_names, train_files, val_files


def make_loader(files, processor, num_frames: int, args, use_ddp: bool, train: bool) -> DataLoader:
    ds = VideoMambaDataset(files, processor, num_frames=num_frames)
    sampler = DistributedSampler(ds, shuffle=train) if use_ddp else None
    # SSv2 WebM files trigger Decord EAGAIN (-11) errors when multiple VideoReader
    # instances run concurrently (FFmpeg internal thread contention). Force
    # num_workers=1 to ensure only one Decord instance at a time.
    ssv2_mode = getattr(args, "dataset", "") == "ssv2"
    nw = 1 if ssv2_mode else args.num_workers
    persistent = nw > 0 and not ssv2_mode
    prefetch = 2 if (ssv2_mode and nw > 0) else (4 if nw > 0 else None)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch if nw > 0 else None,
        multiprocessing_context="forkserver" if nw > 0 else None,
    )


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(args, num_classes: int, train_n: int, val_n: int) -> Optional[object]:
    if args.no_wandb or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb unavailable; skipping")
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"train-videomamba-{Path(args.save_path).name}",
        tags=args.wandb_tags,
        config={
            "dataset": args.dataset,
            "model": "videomamba",
            "architecture": "SSM",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_frames": 8,
            "input_size": 224,
            "class_count": num_classes,
            "train_count": train_n,
            "val_count": val_n,
            "phase": "train",
            "ddp": args.ddp,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def reduce_sum(value: float, device: torch.device) -> float:
    t = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def train_one_epoch(model, loader, optimizer, device, epoch: int, show_progress: bool):
    model.train()
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc=f"epoch={epoch} train", disable=not show_progress)
    for batch in pbar:
        labels       = batch.pop("labels").to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(pixel_values=pixel_values).logits
            loss   = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * labels.numel()
        total_n    += labels.numel()
        if show_progress:
            pbar.set_postfix(loss=total_loss / max(1, total_n))
    total_loss = reduce_sum(total_loss, device)
    total_n    = reduce_sum(total_n, device)
    return total_loss / max(1.0, total_n)


@torch.inference_mode()
def evaluate(model, loader, device, show_progress: bool):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0.0, 0.0
    for batch in tqdm(loader, desc="validation", disable=not show_progress):
        labels       = batch.pop("labels").to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(pixel_values=pixel_values).logits
            loss   = torch.nn.functional.cross_entropy(logits, labels)
        total_loss    += float(loss.item()) * labels.numel()
        total_correct += float((logits.argmax(dim=-1) == labels).sum().item())
        total_n       += float(labels.numel())
    total_loss    = reduce_sum(total_loss, device)
    total_correct = reduce_sum(total_correct, device)
    total_n       = reduce_sum(total_n, device)
    return total_loss / max(1.0, total_n), total_correct / max(1.0, total_n)


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(
    save_dir: str | Path,
    model,
    class_names: list[str],
    extra: dict | None = None,
    input_size: int = 224,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw = model.module if hasattr(model, "module") else model
    raw.save_pretrained(save_dir)  # saves model.pth

    meta = {
        "backend":      "videomamba",
        "model_name":   "videomamba",
        "architecture": "SSM",
        "num_labels":   len(class_names),
        "class_names":  [str(c) for c in class_names],
        "num_frames":   8,
        "input_size":   input_size,
        "embed_dim":    576,
        "depth":        32,
    }
    if extra:
        meta.update(extra)

    with open(save_dir / "accv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Write config.json so eval_fixed_budget.py can detect the backend
    with open(save_dir / "config.json", "w") as f:
        json.dump({"backend": "videomamba"}, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ssv2",
                   choices=list(_DEFAULT_DATA_ROOTS.keys()),
                   help="Dataset to train on (choices auto-populated from _DEFAULT_DATA_ROOTS)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup from 0 to --lr over this many epochs (0=disabled)")
    p.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none",
                   help="LR scheduler after warmup (cosine anneals to 0 over remaining epochs)")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop if val_acc does not improve for this many epochs (0=disabled)")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--save-path", default=None,
                   help="Checkpoint dir (auto-generated as accv2026_videomamba_<dataset> if omitted)")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip K400 pretrained weights (random init)")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint dir to resume training from")
    p.add_argument("--pretrained-from", default=None,
                   help="Fine-tuned checkpoint dir to initialize weights from (epoch resets to 1). "
                        "Use for cross-resolution transfer, e.g. AUTSL 224px → 96px, "
                        "so BiMamba already adapted to target domain.")
    p.add_argument("--input-size", type=int, default=224,
                   help="Spatial resolution for training (default: 224). "
                        "Use for spatial aliasing ablation (e.g. --input-size 112).")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (requires enough VRAM; safe on H200/141GB, may OOM on L40S/46GB)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="inforates-accv2026")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.save_path is None:
        args.save_path = f"fine_tuned_models/accv2026_videomamba_{args.dataset}"
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        torch.cuda.set_device(local_rank)
    device    = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main   = local_rank == 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere/Hopper: ~3× matmul speedup

    class_names, train_files, val_files = prepare_data(
        args.dataset, args.data_root, args.max_train_samples, args.max_val_samples
    )
    if is_main:
        print(f"Classes: {len(class_names)} | Train: {len(train_files)} | Val: {len(val_files)}")

    if is_main:
        print(f"Spatial resolution: {args.input_size}px (native: 224px)")
    processor    = VideoMambaProcessor(size=args.input_size)
    train_loader = make_loader(train_files, processor, num_frames=8, args=args, use_ddp=args.ddp, train=True)
    val_loader   = make_loader(val_files,   processor, num_frames=8, args=args, use_ddp=args.ddp, train=False)

    start_epoch = 1
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path and (resume_path / "accv_meta.json").exists():
        model, _, meta = load_videomamba_checkpoint(str(resume_path), device=str(device))
        start_epoch = meta.get("epoch", 1) + 1
        if is_main:
            print(f"[Resume] from {resume_path}, starting epoch {start_epoch}")
    elif args.pretrained_from:
        model = build_videomamba_from_finetuned(
            checkpoint_dir=args.pretrained_from,
            target_img_size=args.input_size,
        )
        model = model.to(device)
        if is_main:
            print(f"[Pretrained-from] {args.pretrained_from} → {args.input_size}px, training from epoch 1")
    else:
        pretrained = None if args.no_pretrained else PRETRAINED_PATH
        model = build_videomamba(
            num_classes=len(class_names),
            num_frames=8,
            pretrained_path=pretrained,
            img_size=args.input_size,
        )
        model = model.to(device)

    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model._set_static_graph()  # BiMamba reuses params bidirectionally; static graph avoids DDP "ready once" error

    # torch.compile: enabled via --compile flag (needs ≥80GB VRAM; SSM CUDA Graphs ~49GB pool)
    if args.compile and torch.cuda.is_available() and not args.ddp:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            if is_main:
                print("[torch.compile] compiled successfully")
        except Exception as e:
            if is_main:
                print(f"[torch.compile] skipped: {e}")

    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler: optional warmup then cosine annealing
    cosine_epochs = args.epochs - args.warmup_epochs
    if args.lr_scheduler == "cosine" and cosine_epochs > 0:
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=args.lr * 0.01
        )
    else:
        cosine_sched = None

    def _update_lr(epoch: int):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            # Linear warmup: scale from lr/100 to lr
            scale = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * max(scale, 0.01)
        elif cosine_sched is not None and epoch > args.warmup_epochs:
            cosine_sched.step()

    wandb_run  = init_wandb(args, len(class_names), len(train_files), len(val_files)) if is_main else None

    best_acc = -1.0
    best_epoch = start_epoch
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        _update_lr(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        if args.ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_loss          = train_one_epoch(model, train_loader, optimizer, device, epoch, is_main)
        val_loss, val_acc   = evaluate(model, val_loader, device, is_main)

        if is_main:
            print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}")
            if wandb_run is not None:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                           "val_accuracy": val_acc, "lr": current_lr})

            # Per-epoch checkpoints disabled to preserve disk quota

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(args.save_path, model, class_names,
                                extra={"epoch": epoch, "val_acc": val_acc, "is_best": True},
                                input_size=args.input_size)
                print(f"  -> New best! Saved to {args.save_path} (epoch {epoch}, val_acc={val_acc:.4f})")
            else:
                no_improve += 1

        # Broadcast early-stopping decision from rank-0
        if args.ddp:
            import torch.distributed as dist
            stop_tensor = torch.tensor(
                1 if (args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience) else 0,
                device=device
            )
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1
        else:
            should_stop = args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience

        if should_stop:
            if is_main:
                print(f"  [EarlyStopping] No improvement for {args.early_stopping_patience} epochs. Stopping at epoch {epoch}.")
            break

    if is_main:
        print(f"Training complete. Best val_acc={best_acc:.4f} at epoch {best_epoch}. Checkpoint: {args.save_path}")
        if wandb_run is not None:
            import wandb
            wandb.summary["checkpoint_path"] = args.save_path
            wandb.summary["best_val_accuracy"] = best_acc
            wandb_run.finish()

    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
