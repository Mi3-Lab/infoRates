#!/usr/bin/env python3
"""Fine-tune VideoMamba3 (Mamba-3 SSM) on video classification datasets — ACCV 2026.

Mirrors train_videomamba.py but uses BiMamba3 instead of BiMamba-1.
Model building optionally loads compatible pretrained weights (patch_embed,
pos_embed, cls_token, per-block norms) from the K400 VideoMamba checkpoint,
skipping the SSM (mixer) params which have a different architecture.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VIDEOMAMBA3_DIR = ROOT / "experiments" / "videomamba3"
for _p in (ROOT, SRC, VIDEOMAMBA3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from types import SimpleNamespace

from info_rates.data.something import (
    get_class_mapping,
    get_numeric_labels,
    get_train_val_test_manifests,
    list_classes,
)
from info_rates.models.videomamba_model import VideoMambaProcessor
from info_rates.training.ddp import cleanup_ddp, setup_ddp

BAD_VIDEOS_LOG = ROOT / "evaluations/accv2026/logs/bad_videos.tsv"
PRETRAINED_PATH = ROOT / "fine_tuned_models" / "videomamba_pretrained" / "videomamba_m16_k400_f8_res224.pth"
PRETRAINED_BY_SIZE = {
    "middle": ROOT / "fine_tuned_models" / "videomamba_pretrained" / "videomamba_m16_k400_f8_res224.pth",
    "small": ROOT / "fine_tuned_models" / "videomamba_pretrained" / "videomamba_s16_k400_f8_res224.pth",
    "tiny": ROOT / "fine_tuned_models" / "videomamba_pretrained" / "videomamba_t16_k400_f8_res224.pth",
}

_DEFAULT_DATA_ROOTS = {
    "ssv2":          "data/Something_data",
    "ucf101":        "data/UCF101_data",
    "hmdb51":        "data/HMDB51_data",
    "diving48":      "data/Diving48_data",
    "epic_kitchens": "data/EPIC_data",
    "autsl":         "data/AUTSL_data",
    "driveact":      "data/DriveAct_data",
}

_EMBED_DIMS = {"tiny": 192, "small": 384, "middle": 576}
_DEPTHS = {"tiny": 24, "small": 24, "middle": 32}


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class _LogitsOutput:
    __slots__ = ("logits",)
    def __init__(self, logits):
        self.logits = logits


class VideoMamba3Model(nn.Module):
    """VideoMamba3 backbone wrapped to match the eval-benchmark interface."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        num_labels = backbone.head.out_features
        self.config = SimpleNamespace(
            num_labels=num_labels,
            id2label={i: str(i) for i in range(num_labels)},
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> _LogitsOutput:
        return _LogitsOutput(self.backbone(pixel_values))

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        raw = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
        torch.save(raw.state_dict(), save_dir / "model.pth")


def _instantiate_videomamba3_raw(
    num_classes: int,
    num_frames: int = 8,
    input_size: int = 224,
    model_size: str = "middle",
    variant: str = "complex",
    mamba3_impl: str = "auto",
    ssm_cfg: dict | None = None,
    depth: int | None = None,
):
    from videomamba3 import VisionMamba, videomamba3_tiny, videomamba3_small, videomamba3_middle

    factory = {
        "tiny":   videomamba3_tiny,
        "small":  videomamba3_small,
        "middle": videomamba3_middle,
    }
    if model_size not in factory:
        raise ValueError(f"model_size must be one of {list(factory)}, got '{model_size}'")
    if depth is None:
        return factory[model_size](
            num_frames=num_frames,
            img_size=input_size,
            num_classes=num_classes,
            variant=variant,
            mamba3_impl=mamba3_impl,
            ssm_cfg=ssm_cfg or {},
        )
    return VisionMamba(
        img_size=input_size,
        patch_size=16,
        depth=depth,
        embed_dim=_EMBED_DIMS[model_size],
        num_classes=num_classes,
        num_frames=num_frames,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        mamba3_variant=variant,
        mamba3_impl=mamba3_impl,
        ssm_cfg=ssm_cfg or {},
    )


def build_videomamba3(
    num_classes: int,
    num_frames: int = 8,
    input_size: int = 224,
    model_size: str = "middle",
    variant: str = "complex",
    mamba3_impl: str = "auto",
    ssm_cfg: dict | None = None,
    depth: int | None = None,
    pretrained_path: str | Path | None = None,
) -> VideoMamba3Model:
    """Instantiate VideoMamba3 with optional partial-pretrained init.

    Compatible params (patch_embed, pos_embed, cls_token, per-block norms)
    are loaded from the K400 VideoMamba checkpoint. SSM (mixer) params are
    skipped — they have a different architecture in Mamba-3.
    """
    model_raw = _instantiate_videomamba3_raw(
        num_classes=num_classes,
        num_frames=num_frames,
        input_size=input_size,
        model_size=model_size,
        variant=variant,
        mamba3_impl=mamba3_impl,
        ssm_cfg=ssm_cfg,
        depth=depth,
    )

    def _resize_pos_embed(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if value.ndim != 3 or target.ndim != 3 or value.shape[-1] != target.shape[-1]:
            return None
        src_tokens = value.shape[1]
        dst_tokens = target.shape[1]
        has_cls = int(math.isqrt(src_tokens - 1) ** 2 == src_tokens - 1 and math.isqrt(dst_tokens - 1) ** 2 == dst_tokens - 1)
        if has_cls:
            cls_pos, value = value[:, :1], value[:, 1:]
            dst_grid_tokens = dst_tokens - 1
        else:
            cls_pos = None
            dst_grid_tokens = dst_tokens
        src_grid = math.isqrt(value.shape[1])
        dst_grid = math.isqrt(dst_grid_tokens)
        if src_grid * src_grid != value.shape[1] or dst_grid * dst_grid != dst_grid_tokens:
            return None
        x = value.transpose(1, 2).reshape(1, value.shape[-1], src_grid, src_grid)
        x = torch.nn.functional.interpolate(x, size=(dst_grid, dst_grid), mode="bicubic", align_corners=False)
        x = x.reshape(1, value.shape[-1], dst_grid * dst_grid).transpose(1, 2)
        return torch.cat([cls_pos, x], dim=1) if cls_pos is not None else x

    def _resize_temporal_pos(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if value.ndim != 3 or target.ndim != 3 or value.shape[-1] != target.shape[-1]:
            return None
        x = value.transpose(1, 2)
        x = torch.nn.functional.interpolate(x, size=target.shape[1], mode="linear", align_corners=False)
        return x.transpose(1, 2)

    if pretrained_path is not None and Path(pretrained_path).exists():
        ckpt = torch.load(str(pretrained_path), map_location="cpu")
        state = ckpt.get("model", ckpt)
        model_state = model_raw.state_dict()
        loaded, skipped = 0, 0
        resized = 0
        partial = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if ".mixer." in k or k.startswith("head."):
                skipped += 1
                continue
            if k in model_state and model_state[k].shape == v.shape:
                partial[k] = v
                loaded += 1
            elif k == "pos_embed" and k in model_state:
                resized_v = _resize_pos_embed(v, model_state[k])
                if resized_v is not None and resized_v.shape == model_state[k].shape:
                    partial[k] = resized_v
                    loaded += 1
                    resized += 1
                else:
                    skipped += 1
            elif k == "temporal_pos_embedding" and k in model_state:
                resized_v = _resize_temporal_pos(v, model_state[k])
                if resized_v is not None and resized_v.shape == model_state[k].shape:
                    partial[k] = resized_v
                    loaded += 1
                    resized += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        model_state.update(partial)
        model_raw.load_state_dict(model_state, strict=False)

        # Reinitialize the classification head for the new num_classes
        embed_dim = _EMBED_DIMS[model_size]
        model_raw.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(model_raw.head.weight, std=0.02)
        nn.init.zeros_(model_raw.head.bias)

        print(f"[VideoMamba3] Partial pretrained load from {pretrained_path}")
        print(f"[VideoMamba3]   {loaded} params loaded ({resized} resized), {skipped} skipped (SSM+head)")
        if loaded == 0:
            print("[VideoMamba3][WARN] No pretrained tensors matched. Check model_size/input_size/pretrained path.")

    return VideoMamba3Model(model_raw)


def load_videomamba3_checkpoint(
    save_dir: str | Path,
    device: str = "cuda",
) -> tuple[VideoMamba3Model, VideoMambaProcessor, dict]:
    """Load a fine-tuned VideoMamba3 checkpoint."""
    save_dir = Path(save_dir)
    meta = json.loads((save_dir / "accv_meta.json").read_text())

    num_classes = meta["num_labels"]
    num_frames  = meta.get("num_frames", 8)
    input_size  = meta.get("input_size", 224)
    model_size  = meta.get("model_name", "videomamba3_middle").split("_")[-1]
    variant     = meta.get("mamba3_variant", "complex")
    mamba3_impl = meta.get("mamba3_impl", "auto")
    ssm_cfg     = meta.get("ssm_cfg", {})
    depth       = meta.get("depth")

    model_raw = _instantiate_videomamba3_raw(
        num_classes=num_classes,
        num_frames=num_frames,
        input_size=input_size,
        model_size=model_size,
        variant=variant,
        mamba3_impl=mamba3_impl,
        ssm_cfg=ssm_cfg,
        depth=depth,
    )
    state_dict = torch.load(save_dir / "model.pth", map_location="cpu")
    model_raw.load_state_dict(state_dict)
    model_raw.eval()
    model_raw.to(device)

    return VideoMamba3Model(model_raw), VideoMambaProcessor(size=input_size), meta


# ---------------------------------------------------------------------------
# Dataset (identical to train_videomamba.py)
# ---------------------------------------------------------------------------

class VideoMambaDataset(Dataset):
    def __init__(self, files, processor, num_frames=8, max_decode_retries=None):
        self.files = files
        self.processor = processor
        self.num_frames = num_frames
        self.max_decode_retries = max_decode_retries or int(
            os.environ.get("INFORATES_MAX_DECODE_RETRIES", "8")
        )
        self.bad_video_log = os.environ.get("INFORATES_BAD_VIDEO_LOG", str(BAD_VIDEOS_LOG))
        self._reported_bad: set = set()
        self.has_labels = isinstance(self.files[0], tuple) if self.files else False

    def __len__(self):
        return len(self.files)

    def _resolve(self, idx):
        if self.has_labels:
            return self.files[idx]
        path = self.files[idx]
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
        import av
        all_frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                all_frames.append(frame.to_ndarray(format="rgb24"))
        if not all_frames:
            raise RuntimeError(f"No frames decoded: {path}")
        idxs = np.linspace(0, len(all_frames) - 1, self.num_frames).astype(int)
        return np.stack([all_frames[i] for i in idxs])

    def __getitem__(self, idx):
        attempts = min(max(1, self.max_decode_retries), max(1, len(self.files)))
        last_error = None
        for offset in range(attempts):
            path, label = self._resolve((idx + offset) % len(self.files))
            try:
                frames = self._decode(path)
                break
            except Exception as e:
                last_error = e
                self._log_bad(path, e)
        else:
            raise RuntimeError(f"Failed after {attempts} attempts: {last_error}")

        inputs = self.processor([frames[i] for i in range(len(frames))], return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
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


def prepare_data(dataset, data_root, max_train, max_val):
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
        train_files = list(zip(train_df["video_path"].tolist(), train_df["label"].astype(int).tolist()))
        val_files   = list(zip(val_df["video_path"].tolist(),   val_df["label"].astype(int).tolist()))
    else:
        from info_rates.data.datasets import load_dataset
        class_names, train_files, val_files = load_dataset(dataset, root)
        train_files = [(p, l) for p, l in train_files if os.path.exists(p)]
        val_files   = [(p, l) for p, l in val_files   if os.path.exists(p)]

    bad_videos = load_bad_video_paths()
    if bad_videos:
        before = len(train_files)
        train_files = [(p, l) for p, l in train_files if p not in bad_videos]
        val_files   = [(p, l) for p, l in val_files   if p not in bad_videos]
        removed = before - len(train_files)
        if removed:
            print(f"[DataFilter] Excluded {removed} known-bad videos")

    train_files = balanced_limit(train_files, max_train)
    val_files = balanced_limit(val_files, max_val)
    return class_names, train_files, val_files


def balanced_limit(files, limit: int, seed: int = 42):
    """Limit a split without collapsing to the first classes in sorted manifests."""
    if limit <= 0 or limit >= len(files):
        return files
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[tuple[str, int]]] = {}
    for path, label in files:
        by_label.setdefault(int(label), []).append((path, int(label)))
    for label in by_label:
        rng.shuffle(by_label[label])

    selected = []
    labels = sorted(by_label)
    while len(selected) < limit and labels:
        next_labels = []
        for label in labels:
            bucket = by_label[label]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_labels.append(label)
        labels = next_labels
    rng.shuffle(selected)
    return selected


def make_loader(files, processor, num_frames, args, use_ddp, train) -> DataLoader:
    ds = VideoMambaDataset(files, processor, num_frames=num_frames)
    sampler = DistributedSampler(ds, shuffle=train) if use_ddp else None
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        multiprocessing_context="forkserver" if args.num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(args, num_classes, train_n, val_n) -> Optional[object]:
    if args.no_wandb or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"train-videomamba3-{args.mamba3_variant}-{Path(args.save_path).name}",
        tags=args.wandb_tags,
        config={
            "dataset": args.dataset,
            "model": f"videomamba3_{args.model_size}",
            "architecture": "SSM-Mamba3",
            "mamba3_variant": args.mamba3_variant,
            "mamba3_impl": args.mamba3_impl,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "ssm_d_state": args.ssm_d_state,
            "ssm_expand": args.ssm_expand,
            "ssm_headdim": args.ssm_headdim,
            "ssm_mimo_rank": args.ssm_mimo_rank,
            "num_frames": args.num_frames,
            "input_size": args.input_size,
            "depth": args.depth or _DEPTHS[args.model_size],
            "class_count": num_classes,
            "train_count": train_n,
            "val_count": val_n,
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


def train_one_epoch(model, loader, optimizer, device, epoch, show_progress):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * labels.numel()
        total_n    += labels.numel()
        if show_progress:
            pbar.set_postfix(loss=total_loss / max(1, total_n))
    return reduce_sum(total_loss, device) / max(1.0, reduce_sum(total_n, device))


@torch.inference_mode()
def evaluate(model, loader, device, show_progress):
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
    return (
        reduce_sum(total_loss, device) / max(1.0, reduce_sum(total_n, device)),
        reduce_sum(total_correct, device) / max(1.0, reduce_sum(total_n, device)),
    )


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(
    save_dir,
    model,
    class_names,
    model_size,
    variant,
    num_frames=8,
    input_size=224,
    depth=None,
    ssm_cfg=None,
    extra=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw = model.module if hasattr(model, "module") else model
    raw.save_pretrained(save_dir)

    meta = {
        "backend":        "videomamba3",
        "model_name":     f"videomamba3_{model_size}",
        "architecture":   "SSM-Mamba3",
        "mamba3_variant": variant,
        "mamba3_impl":    (extra or {}).get("mamba3_impl", "auto"),
        "num_labels":     len(class_names),
        "class_names":    class_names,
        "num_frames":     num_frames,
        "input_size":     input_size,
        "embed_dim":      _EMBED_DIMS[model_size],
        "depth":          depth or _DEPTHS[model_size],
        "ssm_cfg":        ssm_cfg or {},
    }
    if extra:
        meta.update(extra)

    with open(save_dir / "accv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(save_dir / "config.json", "w") as f:
        json.dump(
            {
                "backend": "videomamba3",
                "mamba3_variant": variant,
                "mamba3_impl": (extra or {}).get("mamba3_impl", "auto"),
                "model_size": model_size,
                "depth": depth or _DEPTHS[model_size],
                "ssm_cfg": ssm_cfg or {},
            },
            f,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ucf101",
                   choices=list(_DEFAULT_DATA_ROOTS))
    p.add_argument("--data-root", default=None)
    p.add_argument("--model-size", default="middle", choices=["tiny", "small", "middle"])
    p.add_argument("--depth", type=int, default=0,
                   help="override default depth; use 6-12 for fast community/dev ablations")
    p.add_argument("--mamba3-variant", default="complex",
                   choices=["trapezoidal", "complex", "mimo"])
    p.add_argument("--mamba3-impl", default="auto", choices=["auto", "official", "reference"],
                   help="auto uses upstream Mamba3 kernels when available, otherwise the PyTorch reference")
    p.add_argument("--ssm-d-state", type=int, default=64)
    p.add_argument("--ssm-expand", type=int, default=2)
    p.add_argument("--ssm-headdim", type=int, default=64)
    p.add_argument("--ssm-mimo-rank", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--save-path", default="fine_tuned_models/videomamba3_ucf101")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true",
                   help="train from scratch (no K400 partial init)")
    p.add_argument("--resume-from", default=None,
                   help="path to a VideoMamba3 checkpoint dir to resume from")
    p.add_argument("--torch-compile", action="store_true",
                   help="compile the model with torch.compile; useful for timing runs, but can add a long first-step compile")
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
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    class_names, train_files, val_files = prepare_data(
        args.dataset, args.data_root, args.max_train_samples, args.max_val_samples
    )

    if is_main:
        print(f"Classes: {len(class_names)} | Train: {len(train_files)} | Val: {len(val_files)}")

    processor    = VideoMambaProcessor(size=args.input_size)
    train_loader = make_loader(train_files, processor, num_frames=args.num_frames, args=args, use_ddp=args.ddp, train=True)
    val_loader   = make_loader(val_files,   processor, num_frames=args.num_frames, args=args, use_ddp=args.ddp, train=False)

    start_epoch = 1
    resume_path = Path(args.resume_from) if args.resume_from else None
    ssm_cfg = {
        "d_state": args.ssm_d_state,
        "expand": args.ssm_expand,
        "headdim": args.ssm_headdim,
        "mimo_rank": args.ssm_mimo_rank,
    }
    if resume_path and (resume_path / "accv_meta.json").exists():
        model, _, meta = load_videomamba3_checkpoint(str(resume_path), device=str(device))
        ssm_cfg = meta.get("ssm_cfg", ssm_cfg)
        start_epoch = meta.get("epoch", 1) + 1
        if is_main:
            print(f"[Resume] from {resume_path}, starting epoch {start_epoch}")
    else:
        pretrained = None if args.no_pretrained else PRETRAINED_BY_SIZE.get(args.model_size, PRETRAINED_PATH)
        model = build_videomamba3(
            num_classes=len(class_names),
            num_frames=args.num_frames,
            input_size=args.input_size,
            model_size=args.model_size,
            variant=args.mamba3_variant,
            mamba3_impl=args.mamba3_impl,
            ssm_cfg=ssm_cfg,
            depth=args.depth or None,
            pretrained_path=pretrained,
        )
        model = model.to(device)

    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if is_main:
        print(f"[VideoMamba3-{args.model_size}][{args.mamba3_variant}] params: "
              f"{sum(p.numel() for p in model.parameters()):,}")

    if args.torch_compile and torch.cuda.is_available() and not args.ddp:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            if is_main:
                print("[torch.compile] compiled successfully")
        except Exception as e:
            if is_main:
                print(f"[torch.compile] skipped: {e}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb_run = init_wandb(args, len(class_names), len(train_files), len(val_files)) if is_main else None
    history_path = Path(args.save_path).with_name(Path(args.save_path).name + "_history.csv")
    if is_main and start_epoch == 1:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "epoch_seconds", "samples_per_second"],
            )
            writer.writeheader()

    best_acc = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        if args.ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        train_loss        = train_one_epoch(model, train_loader, optimizer, device, epoch, is_main)
        val_loss, val_acc = evaluate(model, val_loader, device, is_main)
        epoch_seconds = time.perf_counter() - epoch_start
        samples_per_second = len(train_files) / max(epoch_seconds, 1e-9)

        if is_main:
            print(
                f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"epoch_s={epoch_seconds:.1f}  samples_s={samples_per_second:.3f}"
            )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch_seconds": epoch_seconds,
                    "samples_per_second": samples_per_second,
                })
            with open(history_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "epoch_seconds", "samples_per_second"],
                )
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8f}",
                    "val_loss": f"{val_loss:.8f}",
                    "val_accuracy": f"{val_acc:.8f}",
                    "epoch_seconds": f"{epoch_seconds:.3f}",
                    "samples_per_second": f"{samples_per_second:.8f}",
                })

            epoch_ckpt = f"{args.save_path}_epoch{epoch}"
            save_checkpoint(epoch_ckpt, model, class_names, args.model_size, args.mamba3_variant,
                            args.num_frames, args.input_size, args.depth or None, ssm_cfg,
                            extra={"epoch": epoch, "val_acc": val_acc, "mamba3_impl": args.mamba3_impl})
            print(f"  -> Saved: {epoch_ckpt}")

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(args.save_path, model, class_names, args.model_size, args.mamba3_variant,
                                args.num_frames, args.input_size, args.depth or None, ssm_cfg,
                                extra={"epoch": epoch, "val_acc": val_acc, "is_best": True,
                                       "mamba3_impl": args.mamba3_impl})
                print(f"  -> New best! Saved to {args.save_path} (epoch {epoch}, val_acc={val_acc:.4f})")

    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
