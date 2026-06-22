#!/usr/bin/env python3
"""Fine-tune VideoMamba3-AFA (Adaptive Frame Allocation) on video datasets.

Two-stage training:
  Stage 1 (scanner) + TCH: supervised by L_concentration (E3 targets)
  Stage 2 (classifier):    supervised by L_cls (CrossEntropy)
  Full model:              L_total = L_cls + λ_conc·L_concentration + λ_budget·L_budget

Usage (single GPU):
  python train_videomamba3_afa.py --dataset ucf101 --variant small

Usage (DDP, 2 GPUs):
  torchrun --nproc_per_node=2 train_videomamba3_afa.py --dataset ucf101 --ddp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
VM3  = ROOT / "experiments" / "videomamba3"
E3   = ROOT / "evaluations" / "accv2026" / "e3_spectral"
for _p in (ROOT, SRC, VM3):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from videomamba3_afa import (
    VideoMamba3AFA,
    videomamba3_afa_tiny,
    videomamba3_afa_small,
    videomamba3_afa_base,
)
from afa_module import AFALoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_ROOTS = {
    "ucf101":        "/scratch/wesleyferreiramaia/infoRates/ucf101",
    "hmdb51":        "/scratch/wesleyferreiramaia/infoRates/hmdb51",
    "ssv2":          "/scratch/wesleyferreiramaia/infoRates/something-something-v2",
    "diving48":      "/scratch/wesleyferreiramaia/infoRates/Diving48",
    "autsl":         "/scratch/wesleyferreiramaia/infoRates/AUTSL",
    "driveact":      "/scratch/wesleyferreiramaia/infoRates/DriveAct",
    "epic_kitchens": "/scratch/wesleyferreiramaia/infoRates/epic-kitchens",
    "finegym":       "/scratch/wesleyferreiramaia/infoRates/FineGym",
}

_MANIFESTS = ROOT / "evaluations" / "accv2026" / "manifests"

_MANIFEST_NAMES = {
    "ucf101":        "ucf101_{split}_full.csv",
    "hmdb51":        "hmdb51_{split}_full.csv",
    "ssv2":          "somethingv2_{split}_full.csv",
    "diving48":      "diving48_{split}_full.csv",
    "autsl":         "autsl_{split}_full.csv",
    "driveact":      "driveact_{split}_full.csv",
    "epic_kitchens": "epic_kitchens_{split}_full.csv",
    "finegym":       "finegym_{split}_full.csv",
}

BAD_VIDEOS_LOG = ROOT / "evaluations" / "accv2026" / "bad_videos.txt"


# ---------------------------------------------------------------------------
# Data utilities (reused from train_videomamba3.py)
# ---------------------------------------------------------------------------

def load_bad_video_paths() -> set[str]:
    if BAD_VIDEOS_LOG.exists():
        return {l.strip() for l in BAD_VIDEOS_LOG.read_text().splitlines() if l.strip()}
    return set()


def decode_frames(path: str, n: int) -> Optional[np.ndarray]:
    """Decode n uniformly-spaced frames. Returns [n, H, W, 3] or None."""
    try:
        from decord import VideoReader, cpu as dec_cpu
        vr = VideoReader(str(path), ctx=dec_cpu(0))
        total = len(vr)
        if total < 2:
            return None
        idx = np.linspace(0, total - 1, n).astype(int)
        arr = vr.get_batch(idx).asnumpy()
        return arr
    except Exception:
        pass
    try:
        import av
        frames = []
        with av.open(str(path)) as c:
            for f in c.decode(c.streams.video[0]):
                frames.append(f.to_ndarray(format="rgb24"))
        if len(frames) < 2:
            return None
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return np.stack([frames[i] for i in idx])
    except Exception:
        return None


class VideoAFADataset(Dataset):
    """Returns full T_max frames as a float32 tensor [C, T_max, H, W]."""

    def __init__(
        self,
        files: list[tuple[str, int]],
        T_max: int = 48,
        img_size: int = 224,
        train: bool = True,
        bad_paths: Optional[set] = None,
    ):
        self.files    = files
        self.T_max    = T_max
        self.size     = img_size
        self.train    = train
        self.bad      = bad_paths or set()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        if str(path) in self.bad:
            return self.__getitem__((idx + 1) % len(self))

        frames = decode_frames(path, self.T_max)
        if frames is None:
            _log_bad(path)
            return self.__getitem__((idx + 1) % len(self))

        # resize + augment
        out = []
        for i in range(self.T_max):
            frame = cv2.resize(frames[i], (self.size, self.size))
            if self.train:
                # random horizontal flip
                if np.random.rand() < 0.5:
                    frame = frame[:, ::-1]
            frame = (frame.astype(np.float32) / 255.0 - self.mean) / self.std
            out.append(frame.transpose(2, 0, 1))   # [3, H, W]

        video = torch.from_numpy(np.stack(out))    # [T, 3, H, W]
        video = video.permute(1, 0, 2, 3)          # [3, T, H, W]
        return video, label


_bad_log_lock = None

def _log_bad(path):
    with open(BAD_VIDEOS_LOG, "a") as f:
        f.write(str(path) + "\n")


def prepare_data(dataset: str, data_root: Optional[str], max_train: int, max_val: int):
    import pandas as pd
    root = Path(data_root) if data_root else Path(_DATA_ROOTS[dataset])

    def load_split(split):
        name = _MANIFEST_NAMES[dataset].format(split=split)
        path = _MANIFESTS / name
        if not path.exists():
            # fallback: try val → test
            alt = _MANIFESTS / _MANIFEST_NAMES[dataset].format(split="test")
            if alt.exists():
                path = alt
            else:
                return [], []
        df = pd.read_csv(path)
        if "exists" in df.columns:
            df = df[df["exists"] == True]
        files = [(str(root / r.video_path), int(r.label_id)) for _, r in df.iterrows()]
        return files, sorted(df["label_name"].unique().tolist() if "label_name" in df.columns else [])

    train_files, class_names = load_split("train")
    val_files, _            = load_split("val")

    if not class_names:
        all_labels = sorted({lab for _, lab in train_files + val_files})
        class_names = [str(c) for c in all_labels]

    if max_train > 0:
        train_files = train_files[:max_train]
    if max_val > 0:
        val_files = val_files[:max_val]

    return class_names, train_files, val_files


def make_loader(files, T_max, img_size, args, use_ddp: bool, train: bool) -> DataLoader:
    bad = load_bad_video_paths()
    ds = VideoAFADataset(files, T_max=T_max, img_size=img_size, train=train, bad_paths=bad)
    sampler = DistributedSampler(ds, shuffle=train) if use_ddp else None
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=(args.num_workers > 0),
    )


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp(local_rank: int):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def reduce_sum(value: float, device) -> float:
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_VARIANTS = {
    "tiny":  videomamba3_afa_tiny,
    "small": videomamba3_afa_small,
    "base":  videomamba3_afa_base,
}


def build_model(args, num_classes: int) -> VideoMamba3AFA:
    factory = _VARIANTS[args.variant]
    model = factory(
        num_classes=num_classes,
        drop_path_rate=args.drop_path_rate,
        selector_temperature=args.selector_temperature,
    )
    return model


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(args, num_classes, train_n, val_n):
    try:
        import wandb
        run_name = args.wandb_run_name or f"afa-{args.variant}-{args.dataset}"
        tags = (args.wandb_tags or []) + ["afa", args.variant, args.dataset]
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=tags,
            config={
                "dataset":             args.dataset,
                "variant":             args.variant,
                "num_classes":         num_classes,
                "T_max":               args.T_max,
                "budget_B":            args.budget_B,
                "epochs":              args.epochs,
                "batch_size":          args.batch_size,
                "lr":                  args.lr,
                "weight_decay":        args.weight_decay,
                "lambda_conc":         args.lambda_conc,
                "lambda_budget":       args.lambda_budget,
                "train_samples":       train_n,
                "val_samples":         val_n,
                "selector_temperature": args.selector_temperature,
            },
        )
        return wandb
    except Exception as e:
        print(f"[W&B] disabled: {e}")
        return None


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, epoch, dataset, is_main, wandb_run
):
    model.train()
    total_loss = total_cls = total_conc = total_budget = 0.0
    correct = n = 0

    for step, (videos, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, scores = model(videos, return_scores=True)
        loss, breakdown = criterion(logits, labels, scores, dataset)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss   += breakdown["loss_total"]  * bs
        total_cls    += breakdown["loss_cls"]    * bs
        total_conc   += breakdown["loss_conc"]   * bs
        total_budget += breakdown["loss_budget"] * bs
        correct      += (logits.argmax(1) == labels).sum().item()
        n            += bs

        if is_main and (step + 1) % 50 == 0:
            print(f"  step {step+1}/{len(loader)} | "
                  f"loss={total_loss/n:.4f} acc={100*correct/n:.1f}%")

    metrics = {
        "train/loss":        total_loss  / n,
        "train/loss_cls":    total_cls   / n,
        "train/loss_conc":   total_conc  / n,
        "train/loss_budget": total_budget / n,
        "train/acc":         correct     / n,
        "epoch":             epoch,
    }
    if wandb_run and is_main:
        wandb_run.log(metrics)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, dataset, is_main, wandb_run):
    model.eval()
    total_loss = correct = n = 0

    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, scores = model(videos, return_scores=True)
        loss, breakdown = criterion(logits, labels, scores, dataset)
        bs = labels.size(0)
        total_loss += breakdown["loss_total"] * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs

    metrics = {
        "val/loss": total_loss / n,
        "val/acc":  correct    / n,
        "epoch":    epoch,
    }
    if wandb_run and is_main:
        wandb_run.log(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    save_path: Path,
    epoch: int,
    val_acc: float,
    args,
    num_classes: int,
):
    save_path.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    torch.save(raw.state_dict(), save_path / "pytorch_model.bin")
    meta = {
        "epoch":      epoch,
        "val_acc":    val_acc,
        "num_classes": num_classes,
        "variant":    args.variant,
        "dataset":    args.dataset,
        "T_max":      args.T_max,
        "budget_B":   args.budget_B,
        "img_size":   args.img_size,
    }
    (save_path / "accv_meta.json").write_text(json.dumps(meta, indent=2))
    # HF-compatible config
    (save_path / "config.json").write_text(json.dumps({
        "model_type":  "videomamba3_afa",
        "num_classes": num_classes,
        **meta,
    }, indent=2))


def load_checkpoint(model: nn.Module, ckpt_dir: Path, device):
    weights = ckpt_dir / "pytorch_model.bin"
    if not weights.exists():
        raise FileNotFoundError(f"No pytorch_model.bin in {ckpt_dir}")
    state = torch.load(weights, map_location=device)
    raw = model.module if isinstance(model, DDP) else model
    missing, unexpected = raw.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [ckpt] unexpected keys: {unexpected[:5]}")
    meta = {}
    mp = ckpt_dir / "accv_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return meta


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # dataset
    p.add_argument("--dataset", default="ucf101", choices=list(_DATA_ROOTS))
    p.add_argument("--data-root", default=None)
    # model
    p.add_argument("--variant", default="small", choices=["tiny", "small", "base"])
    p.add_argument("--T-max", dest="T_max", type=int, default=48)
    p.add_argument("--budget-B", dest="budget_B", type=int, default=None,
                   help="override budget_B from variant default")
    p.add_argument("--img-size", dest="img_size", type=int, default=224)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--selector-temperature", type=float, default=1.0)
    # loss
    p.add_argument("--lambda-conc", type=float, default=0.5)
    p.add_argument("--lambda-budget", type=float, default=0.1)
    p.add_argument("--e3-dir", default=str(E3))
    # training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--resume-from", default=None)
    # infra
    p.add_argument("--save-path", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--ddp", action="store_true")
    # W&B
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="inforates-accv2026")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        setup_ddp(local_rank)
        device  = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    class_names, train_files, val_files = prepare_data(
        args.dataset, args.data_root, args.max_train_samples, args.max_val_samples
    )
    num_classes = len(class_names)

    if is_main:
        print(f"Dataset: {args.dataset} | classes: {num_classes} | "
              f"train: {len(train_files)} | val: {len(val_files)}")

    # build model
    model = build_model(args, num_classes).to(device)
    T_max    = model.T_max
    budget_B = model.budget_B

    # override budget_B if requested
    if args.budget_B and args.budget_B != budget_B:
        print(f"[WARN] --budget-B {args.budget_B} != variant default {budget_B}; "
              f"to change budget you must modify the variant constructor directly.")

    if is_main:
        params = model.n_params
        print(f"VideoMamba3-AFA-{args.variant} | "
              f"Stage1: {params['stage1']/1e6:.1f}M | "
              f"TCH: {params['tch']/1e3:.0f}K | "
              f"Stage2: {params['stage2']/1e6:.1f}M | "
              f"Total: {params['total']/1e6:.1f}M")
        print(f"T_max={T_max}, budget_B={budget_B} "
              f"({100*budget_B/T_max:.0f}% of frames)")

    # resume
    start_epoch = 1
    if args.resume_from:
        meta = load_checkpoint(model, Path(args.resume_from), device)
        start_epoch = meta.get("epoch", 1) + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch - 1}, "
                  f"val_acc={meta.get('val_acc', '?'):.4f}")

    if args.ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # data
    train_loader = make_loader(train_files, T_max, args.img_size, args, args.ddp, train=True)
    val_loader   = make_loader(val_files,   T_max, args.img_size, args, args.ddp, train=False)

    # loss
    criterion = AFALoss(
        e3_dir=args.e3_dir,
        datasets=[args.dataset],
        budget_B=budget_B,
        T_max=T_max,
        lambda_conc=args.lambda_conc,
        lambda_budget=args.lambda_budget,
    )

    # optimizer + cosine schedule with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # W&B
    wandb_run = None
    if is_main and not args.no_wandb:
        wandb_run = init_wandb(args, num_classes, len(train_files), len(val_files))

    save_path = Path(args.save_path) if args.save_path else (
        Path("/scratch/wesleyferreiramaia/infoRates/fine_tuned_models") /
        f"accv2026_videomamba3_afa_{args.variant}_{args.dataset}_e{args.epochs}_h200"
    )

    best_acc  = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.dataset, is_main, wandb_run,
        )
        val_m = evaluate(
            model, val_loader, criterion,
            device, epoch, args.dataset, is_main, wandb_run,
        )
        scheduler.step()

        if is_main:
            elapsed = time.time() - t0
            val_acc = val_m["val/acc"]
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"train_loss={train_m['train/loss']:.4f} "
                  f"train_acc={100*train_m['train/acc']:.2f}% | "
                  f"val_loss={val_m['val/loss']:.4f} "
                  f"val_acc={100*val_acc:.2f}% | "
                  f"conc_loss={train_m['train/loss_conc']:.4f} | "
                  f"{elapsed:.0f}s")

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, save_path, epoch, val_acc, args, num_classes)
                print(f"  ✓ Best checkpoint saved ({100*best_acc:.2f}%)")

            # per-epoch checkpoint
            if args.save_every > 0 and epoch % args.save_every == 0:
                ep_path = save_path.parent / f"{save_path.name}_epoch{epoch}"
                save_checkpoint(model, ep_path, epoch, val_acc, args, num_classes)

            if wandb_run:
                wandb_run.log({"best_val_acc": best_acc, "epoch": epoch})

    if is_main:
        print(f"\nTraining complete. Best val acc: {100*best_acc:.2f}%")
        print(f"Checkpoint: {save_path}")
        if wandb_run:
            wandb_run.finish()

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
