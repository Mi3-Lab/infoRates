"""Training loop for EVA-Mamba3.

Usage:
    TORCH_CUDA_ARCH_LIST="12.0" python src/eva_mamba3/train.py \\
        --dataset finegym --n_classes 97 --epochs 50 --batch_size 64

Phase robustness loss requires two forward passes with different temporal
offsets of the same clip. We implement this by decoding offset frames on-the-fly
from the second copy in the batch (see --phase_loss flag).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from .datasets import build_dataloader
from .factory  import BACKBONE_REGISTRY
from .losses   import EVALoss
from .wrapper  import EVAWrapper

STRIDES = [1, 2, 4, 8, 16]
ROOT    = Path(__file__).resolve().parents[2]


def configure_accelerator() -> None:
    """Enable fast kernels on modern NVIDIA GPUs.

    Blackwell/Ada/Hopper class cards benefit from TF32 matmuls for fp32 paths,
    while the training loop itself uses bf16 autocast on CUDA.
    """
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_timesformer_encoder_init(model: nn.Module, ckpt_path: str | Path) -> None:
    """Initialise an EVA TimeSformer frame encoder from a fine-tuned baseline.

    The clean TimeSformer baseline stores its backbone under
    ``model.timesformer.*``. EVA's frame encoder stores the same module under
    ``frame_encoder.model.*``. We intentionally copy only matching backbone
    tensors and leave EVA/tokenizer/temporal-head parameters randomly
    initialised.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"TimeSformer init checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    source = ckpt.get("model", ckpt)
    target = model.state_dict()
    mapped = {}
    skipped_shape = []

    for key, value in source.items():
        if key.startswith("model.timesformer."):
            dst = "frame_encoder.model." + key[len("model.timesformer."):]
        elif key in {"mean", "std"}:
            dst = "frame_encoder." + key
        else:
            continue

        if dst in target and target[dst].shape == value.shape:
            mapped[dst] = value
        elif dst in target:
            skipped_shape.append((key, tuple(value.shape), tuple(target[dst].shape)))

    if not mapped:
        raise RuntimeError(
            f"No compatible TimeSformer encoder weights were loaded from {path}"
        )

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(
        f"Loaded {len(mapped)} TimeSformer encoder tensors from {path} "
        f"(skipped_shape={len(skipped_shape)}, missing_model_tensors={len(missing)}, "
        f"unexpected={len(unexpected)})",
        flush=True,
    )
    if skipped_shape:
        preview = ", ".join(
            f"{name}: {src}->{dst}" for name, src, dst in skipped_shape[:3]
        )
        print(f"Shape-skipped TimeSformer tensors: {preview}", flush=True)


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch: int, n_epochs: int, lr: float, warmup: int = 5):
    if epoch < warmup:
        lr_scale = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (n_epochs - warmup)
        lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * lr_scale


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      EVAWrapper,
    loader,
    optimizer:  torch.optim.Optimizer,
    criterion:  EVALoss,
    scaler:     torch.amp.GradScaler,
    epoch:      int,
    n_epochs:   int,
    device:     torch.device,
    phase_loss: bool = True,
    train_strides: list[int] | None = None,
    warmup_strides: list[int] | None = None,
    stride_warmup_epochs: int = 0,
    max_batches: int | None = None,
    log_every_batches: int = 50,
    progress_path: Path | None = None,
) -> dict:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    train_strides = train_strides or STRIDES
    active_strides = (
        warmup_strides or [1]
        if epoch < stride_warmup_epochs
        else train_strides
    )
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)   # (B, T, H, W, 3)
        labels = labels.to(device, non_blocking=True)

        # Sample only from the configured training strides. Baselines should
        # usually train with [1]; EVA/robustness methods may train multi-stride.
        s = random.choice(active_strides)

        with amp_context(device):
            out = model(frames, s=s)

            # Phase-shifted forward (random offset within [0, s-1])
            out_phase2 = None
            if phase_loss and s > 1:
                phi = random.randint(1, s - 1)
                frames_shifted = torch.roll(frames, shifts=-phi, dims=1)
                out_phase2 = model(frames_shifted, s=s)

            loss_dict = criterion(out, labels, s, out_phase2)

        scaler.scale(loss_dict["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1
        if (
            n_batches == 1
            or n_batches == total_batches
            or n_batches % log_every_batches == 0
        ):
            avg = {k: v / n_batches for k, v in totals.items()}
            msg = (
                f"[epoch {epoch + 1}/{n_epochs} batch {n_batches}/{total_batches}] "
                f"loss={avg['total']:.3f}"
            )
            print(msg, flush=True)
            if progress_path is not None:
                with open(progress_path, "w") as f:
                    json.dump({
                        "stage": "train_batch",
                        "epoch": epoch,
                        "epochs": n_epochs,
                        "batch": n_batches,
                        "batches": total_batches,
                        "active_strides": active_strides,
                        "train_strides": train_strides,
                        "train_losses": {k: float(v) for k, v in avg.items()},
                    }, f, indent=2, sort_keys=True)
        if max_batches is not None and n_batches >= max_batches:
            break

    return {k: v / n_batches for k, v in totals.items()}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:  EVAWrapper,
    loader,
    device: torch.device,
    strides = STRIDES,
    max_batches: int | None = None,
) -> dict[int, float]:
    model.eval()
    results = {}
    for s in strides:
        correct = total = 0
        for batch_idx, (frames, labels) in enumerate(loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_context(device):
                out = model(frames, s=s)
            pred = out["logits"].argmax(-1)
            correct += (pred == labels).sum().item()
            total   += len(labels)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
        results[s] = correct / total if total > 0 else 0.0
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("EVA-Mamba3 training")
    parser.add_argument("--dataset",     default="finegym")
    parser.add_argument("--n_classes",   type=int, default=97)
    parser.add_argument("--backbone",    default="mamba3",
                        choices=list(BACKBONE_REGISTRY),
                        help="Recommended start: timesformer_baseline, then eva_timesformer")
    parser.add_argument("--model_name",  default=None,
                        help="HF model id/path for TimeSformer-based backbones")
    parser.add_argument("--init_timesformer_encoder", default=None,
                        help="Checkpoint from a fine-tuned timesformer_baseline run; "
                             "copies model.timesformer.* into EVA's frame encoder.")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--wd",          type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--T",           type=int, default=64,
                        help="Dense frames decoded per clip")
    parser.add_argument("--data_root",   default=None,
                        help="Override dataset root")
    parser.add_argument("--train_split", default="train", choices=["train", "val"],
                        help="Use val for local dry-runs when train videos are unavailable")
    parser.add_argument("--target_events", type=float, default=1.0,
                        help="Target number of event frames per EVA window")
    parser.add_argument("--train_strides", nargs="+", type=int, default=None,
                        help="Strides sampled during training. Use '1' for a clean baseline; "
                             "use '1 2 4 8 16' for EVA/robust training.")
    parser.add_argument("--stride_warmup_epochs", type=int, default=0,
                        help="Train with --warmup_strides for this many epochs before "
                             "switching to --train_strides.")
    parser.add_argument("--warmup_strides", nargs="+", type=int, default=[1],
                        help="Strides used during stride warmup.")
    parser.add_argument("--robust_tds_weight", type=float, default=0.5,
                        help="Checkpoint score: mean_acc - robust_tds_weight * TDS.")
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Debug: stop each epoch after N train batches")
    parser.add_argument("--max_val_batches", type=int, default=None,
                        help="Debug: stop validation after N batches per stride")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--log_every_batches", type=int, default=50,
                        help="Print/write training progress every N batches")
    parser.add_argument("--unfreeze_at", type=int, default=10,
                        help="Epoch to unfreeze the frame encoder")
    parser.add_argument("--resume", default=None,
                        help="Resume training from a checkpoint saved by this script")
    parser.add_argument("--no_phase_loss", action="store_true")
    parser.add_argument("--output", default="checkpoints/eva_mamba3/")
    args = parser.parse_args()
    configure_accelerator()
    train_strides = args.train_strides or STRIDES
    invalid_strides = sorted(set(train_strides) - set(STRIDES))
    if invalid_strides:
        raise ValueError(f"Unsupported train strides: {invalid_strides}. Allowed: {STRIDES}")
    invalid_warmup = sorted(set(args.warmup_strides) - set(STRIDES))
    if invalid_warmup:
        raise ValueError(f"Unsupported warmup strides: {invalid_warmup}. Allowed: {STRIDES}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output / args.backbone / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = build_dataloader(
        args.dataset, split=args.train_split,
        batch_size=args.batch_size, num_workers=args.num_workers,
        T=args.T, data_root=args.data_root,
    )
    val_loader = build_dataloader(
        args.dataset, split="val",
        batch_size=max(1, args.batch_size // 2), num_workers=args.num_workers,
        augment=False, T=args.T, data_root=args.data_root,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    build_fn = BACKBONE_REGISTRY[args.backbone]
    if args.model_name:
        model: EVAWrapper = build_fn(
            n_classes=args.n_classes,
            model_name=args.model_name,
        ).to(device)
    else:
        model: EVAWrapper = build_fn(n_classes=args.n_classes).to(device)
    if args.init_timesformer_encoder:
        if not hasattr(model, "frame_encoder"):
            raise ValueError("--init_timesformer_encoder only applies to EVA-style models")
        load_timesformer_encoder_init(model, args.init_timesformer_encoder)
    criterion = EVALoss(target_events=args.target_events)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_tds = float("inf")
    best_mean_acc = -1.0
    best_robust_score = -float("inf")
    start_epoch = 0
    progress_path = out_dir / "progress.json"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if "tds" in ckpt:
            best_tds = float(ckpt["tds"])
        if "mean_acc" in ckpt:
            best_mean_acc = float(ckpt["mean_acc"])
        if "robust_score" in ckpt:
            best_robust_score = float(ckpt["robust_score"])
        best_acc_path = out_dir / "best_acc.pt"
        if best_acc_path.exists():
            best_ckpt = torch.load(best_acc_path, map_location="cpu")
            if "mean_acc" in best_ckpt:
                best_mean_acc = max(best_mean_acc, float(best_ckpt["mean_acc"]))
        best_robust_path = out_dir / "best_robust.pt"
        if best_robust_path.exists():
            best_ckpt = torch.load(best_robust_path, map_location="cpu")
            if "robust_score" in best_ckpt:
                best_robust_score = max(best_robust_score, float(best_ckpt["robust_score"]))
        best_tds_path = out_dir / "best_tds.pt"
        if best_tds_path.exists():
            best_ckpt = torch.load(best_tds_path, map_location="cpu")
            if "tds" in best_ckpt:
                best_tds = min(best_tds, float(best_ckpt["tds"]))
        print(
            f"Resumed from {resume_path} at epoch {start_epoch + 1}/{args.epochs}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        cosine_lr(optimizer, epoch, args.epochs, args.lr)

        # Unfreeze encoder at specified epoch
        if epoch == args.unfreeze_at:
            print(f"[epoch {epoch}] Unfreezing spatial encoder")
            model.unfreeze_encoder()
            # Re-initialise optimizer to include encoder params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.wd,
            )

        t0 = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            epoch=epoch,
            n_epochs=args.epochs,
            device=device,
            phase_loss=not args.no_phase_loss,
            train_strides=train_strides,
            warmup_strides=args.warmup_strides,
            stride_warmup_epochs=args.stride_warmup_epochs,
            max_batches=args.max_train_batches,
            log_every_batches=args.log_every_batches,
            progress_path=progress_path,
        )
        dt = time.time() - t0
        progress = {
            "epoch": epoch,
            "epochs": args.epochs,
            "stage": "train_epoch_done",
            "train_losses": {k: float(v) for k, v in train_losses.items()},
            "epoch_seconds": dt,
            "active_strides": args.warmup_strides if epoch < args.stride_warmup_epochs else train_strides,
            "train_strides": train_strides,
            "stride_warmup_epochs": args.stride_warmup_epochs,
            "args": vars(args),
        }

        should_validate = (epoch % args.val_every == 0) or (epoch == args.epochs - 1)
        if should_validate:
            acc = validate(model, val_loader, device, max_batches=args.max_val_batches)
            tds = max(0.0, acc[1] - acc[16])
            mean_acc = sum(acc.values()) / len(acc)
            robust_score = mean_acc - args.robust_tds_weight * tds
            progress.update({
                "stage": "validation_done",
                "acc": {str(k): float(v) for k, v in acc.items()},
                "tds": float(tds),
                "mean_acc": float(mean_acc),
                "robust_score": float(robust_score),
            })
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2, sort_keys=True)
            torch.save({
                "epoch":  epoch,
                "model":  model.state_dict(),
                "optim":  optimizer.state_dict(),
                "acc":    acc,
                "tds":    tds,
                "mean_acc": mean_acc,
                "robust_score": robust_score,
                "train_losses": train_losses,
                "args":   vars(args),
            }, out_dir / "last.pt")
            acc_msg = "  ".join(f"acc@s{s}={acc[s]*100:.1f}" for s in STRIDES)
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={train_losses['total']:.3f}  "
                f"{acc_msg}  "
                f"TDS={tds*100:.1f}pp  score={robust_score*100:.1f}  ({dt:.0f}s)"
            , flush=True)
            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                ckpt = out_dir / "best_acc.pt"
                payload = {
                    "epoch":  epoch,
                    "model":  model.state_dict(),
                    "optim":  optimizer.state_dict(),
                    "acc":    acc,
                    "tds":    tds,
                    "mean_acc": mean_acc,
                    "args":   vars(args),
                }
                torch.save(payload, ckpt)
                print(f"  → saved best accuracy checkpoint (mean_acc={mean_acc*100:.1f}%)", flush=True)
            if robust_score > best_robust_score:
                best_robust_score = robust_score
                ckpt = out_dir / "best_robust.pt"
                payload = {
                    "epoch":  epoch,
                    "model":  model.state_dict(),
                    "optim":  optimizer.state_dict(),
                    "acc":    acc,
                    "tds":    tds,
                    "mean_acc": mean_acc,
                    "robust_score": robust_score,
                    "args":   vars(args),
                }
                torch.save(payload, ckpt)
                torch.save(payload, out_dir / "best.pt")
                print(f"  → saved best robust checkpoint (score={robust_score*100:.1f})", flush=True)
            if tds < best_tds:
                best_tds = tds
                ckpt = out_dir / "best_tds.pt"
                torch.save({
                    "epoch":  epoch,
                    "model":  model.state_dict(),
                    "optim":  optimizer.state_dict(),
                    "acc":    acc,
                    "tds":    tds,
                    "mean_acc": mean_acc,
                    "robust_score": robust_score,
                    "args":   vars(args),
                }, ckpt)
                print(f"  → saved best TDS checkpoint (TDS={tds*100:.1f}pp)", flush=True)
        else:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2, sort_keys=True)
            torch.save({
                "epoch":  epoch,
                "model":  model.state_dict(),
                "optim":  optimizer.state_dict(),
                "train_losses": train_losses,
                "args":   vars(args),
            }, out_dir / "last.pt")
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={train_losses['total']:.3f}  ({dt:.0f}s)"
            , flush=True)

    print(
        f"\nDone. Best mean acc: {best_mean_acc*100:.1f}%. "
        f"Best robust score: {best_robust_score*100:.1f}. "
        f"Best TDS: {best_tds*100:.1f}pp",
        flush=True,
    )


if __name__ == "__main__":
    main()
