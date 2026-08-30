#!/usr/bin/env python3
"""Multi-stride augmentation ablation (reviewer XdvJ).

XdvJ's central doubt, and the reason they hold at weak accept:

    "Without temporal stride or multi-rate augmentation during fine-tuning,
     models may be inherently biased toward their native training sampling
     scheme. This makes me question whether the observed 'aliasing cliffs'
     reflect intrinsic architectural limitations or, at least partly, robustness
     to a training-inference sampling distribution shift."

A fair question, and the answer is sharper than it looks once you know what the
evaluation sampler actually does. `select_frame_indices` re-uniformizes the
candidate pool, so stride only bites when ceil(window/s) < budget, and it then
pads by repeating the LAST candidate frame. At stride 16 on AUTSL every clip is
in that regime: the model receives ~4 distinct frames and a frozen tail. No
training recipe makes a model extract 16 frames of evidence from 4.

So the ablation has two arms and they are expected to diverge:

  --arm paper   augment using the published evaluation sampler, padding and all.
                Tests XdvJ's hypothesis directly: is the published cliff
                trainable? Prediction: largely NOT, because the cliff is an
                input-degeneracy artifact rather than a distribution shift.

  --arm fixed   augment using uniform resampling, no frozen tail. Tests the
                question the paper should have asked: does multi-rate exposure
                help when the input stays well-formed? Prediction: yes, modestly.

The baseline arm needs no training -- the published fine-tuned checkpoints are
already it.

An earlier attempt at this experiment exists in docs/legacy/. Do not reuse its
numbers: it ran on the pre-fix sampler where `frames[:n_keep:stride]` collapsed
several (coverage, stride) cells to a single frame, which is why its table has
identical values at 25%/s8, 25%/s16 and 50%/s16, and why its headline "+15%
improvement" compares two models that were both being shown one frame.

Usage:
    python scripts/accv2026/train_with_tra.py \
        --model timesformer --dataset autsl --arm paper --epochs 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decord import VideoReader, cpu as decord_cpu  # noqa: E402

from info_rates.evaluation.benchmark import select_frame_indices  # noqa: E402
from info_rates.models.timesformer import UCFDataset  # noqa: E402

# Same grid the paper sweeps, so training sees exactly the conditions it is
# later evaluated on.
COVERAGES = (10, 25, 50, 75, 100)
STRIDES = (1, 2, 4, 8, 16)


def corrected_indices(total: int, budget: int, coverage: int, stride: int) -> np.ndarray:
    """Strided candidates resampled uniformly to `budget` -- never repeat-last."""
    window = max(1, int(round(total * coverage / 100.0)))
    cand = np.arange(0, window, max(1, stride), dtype=np.int64)
    if len(cand) == 0:
        cand = np.array([0], dtype=np.int64)
    pick = np.linspace(0, len(cand) - 1, budget).round().astype(np.int64)
    return cand[pick]


class TRADataset(UCFDataset):
    """UCFDataset that samples a random (coverage, stride) per training item."""

    def __init__(self, *args, arm: str = "paper", p_augment: float = 0.5,
                 seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.p_augment = p_augment
        self.rng = np.random.default_rng(seed)

    def _sample_indices(self, total: int) -> np.ndarray:
        if self.rng.random() >= self.p_augment:
            # Unaugmented item: the standard uniform-over-clip sampling.
            return np.linspace(0, total - 1, self.num_frames).astype(np.int64)
        cov = int(self.rng.choice(COVERAGES))
        stride = int(self.rng.choice(STRIDES))
        if self.arm == "paper":
            return select_frame_indices(total, self.num_frames, cov, stride)
        return corrected_indices(total, self.num_frames, cov, stride)

    def _decode_frames(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        try:
            vr = VideoReader(str(path), ctx=decord_cpu(0))
            total = len(vr)
            if total <= 0:
                raise RuntimeError(f"Video has 0 frames: {path}")
            idxs = self._sample_indices(total)
            # get_batch needs sorted unique positions; padding repeats indices.
            uniq = sorted(set(int(i) for i in idxs))
            raw = vr.get_batch(uniq).asnumpy()
            pos = {p: i for i, p in enumerate(uniq)}
            return np.stack([raw[pos[int(i)]] for i in idxs])
        except FileNotFoundError:
            raise
        except Exception:
            import av
            frames = []
            with av.open(str(path)) as container:
                for frame in container.decode(container.streams.video[0]):
                    frames.append(frame.to_ndarray(format="rgb24"))
            if not frames:
                raise RuntimeError(f"No frames decoded from: {path}")
            idxs = self._sample_indices(len(frames))
            return np.stack([frames[int(i)] for i in idxs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", choices=["paper", "fixed"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--p-augment", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # The per-family training scripts own data prep, the train/eval loop and
    # checkpointing; only the sampler differs here, so reuse rather than fork.
    # They disagree on make_loader's signature, hence the two wrappers below.
    CNNS = {"r3d_18", "mc3_18", "r2plus1d_18"}
    if args.model in CNNS:
        import train_torchvision as tt
        entry = "train_torchvision.py"
    else:
        import train_transformers as tt
        entry = "train_transformers.py"

    save = args.save_path or str(
        ROOT / "fine_tuned_models" / f"accv2026_{args.model}_{args.dataset}_tra_{args.arm}")

    argv = [
        entry,
        "--model", args.model,
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num-workers", str(args.num_workers),
        "--save-path", save,
        "--wandb-run-name", f"tra-{args.arm}-{args.model}-{args.dataset}",
        "--wandb-tags", "tra-ablation", f"arm-{args.arm}",
    ]
    if args.no_wandb:
        argv.append("--no-wandb")

    arm, p_aug, seed = args.arm, args.p_augment, args.seed
    original_loader = tt.make_loader

    def build(files, processor, num_frames, input_size, a):
        from torch.utils.data import DataLoader
        ds = TRADataset(files, processor, num_frames=num_frames, size=input_size,
                        arm=arm, p_augment=p_aug, seed=seed)
        return DataLoader(
            ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
            pin_memory=False, persistent_workers=a.num_workers > 0,
            prefetch_factor=4 if a.num_workers > 0 else None,
            multiprocessing_context="spawn" if a.num_workers > 0 else None,
        )

    if args.model in CNNS:
        # train_torchvision.make_loader(files, processor, args, use_ddp, train)
        def make_loader(files, processor, a, use_ddp, train):
            if not train:
                return original_loader(files, processor, a, use_ddp, train)
            return build(files, processor, a.num_frames, a.input_size, a)
    else:
        # train_transformers.make_loader(files, processor, num_frames, input_size,
        #                                args, use_ddp, train)
        def make_loader(files, processor, num_frames, input_size, a, use_ddp, train):
            if not train:
                return original_loader(files, processor, num_frames, input_size,
                                       a, use_ddp, train)
            return build(files, processor, num_frames, input_size, a)

    tt.make_loader = make_loader
    sys.argv = argv
    print(f"=== TRA ablation: arm={arm}  p_augment={p_aug}  model={args.model} "
          f"-> {save} ===", flush=True)
    tt.main()


if __name__ == "__main__":
    main()
