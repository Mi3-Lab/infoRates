#!/usr/bin/env python3
"""Minimal VideoMamba3 inference example for exported Hugging Face folders."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def decode_uniform_frames(path: Path, num_frames: int) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No video frames decoded from {path}")
    idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [frames[i] for i in idxs]


def preprocess(frames: list[np.ndarray], size: int) -> torch.Tensor:
    tensors = []
    for frame in frames:
        if frame.shape[0] != size or frame.shape[1] != size:
            frame = cv2.resize(frame, (size, size))
        x = torch.from_numpy(frame.copy()).float().div(255.0).permute(2, 0, 1)
        tensors.append((x - IMAGENET_MEAN) / IMAGENET_STD)
    return torch.stack(tensors, dim=1).unsqueeze(0)


def load_model(model_dir: Path, device: torch.device):
    sys.path.insert(0, str(model_dir))
    from videomamba3 import VisionMamba  # noqa: PLC0415

    meta = json.loads((model_dir / "accv_meta.json").read_text())
    model = VisionMamba(
        img_size=meta["input_size"],
        patch_size=16,
        depth=meta["depth"],
        embed_dim=meta["embed_dim"],
        num_classes=meta["num_labels"],
        num_frames=meta["num_frames"],
        mamba3_variant=meta["mamba3_variant"],
        ssm_cfg=meta["ssm_cfg"],
    )
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    return model, meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, meta = load_model(model_dir, device)
    frames = decode_uniform_frames(Path(args.video), meta["num_frames"])
    pixel_values = preprocess(frames, meta["input_size"]).to(device)
    logits = model(pixel_values)
    probs = logits.softmax(dim=-1)[0]
    top = probs.topk(min(args.top_k, probs.numel()))
    labels = meta.get("class_names") or [str(i) for i in range(probs.numel())]
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{idx:4d} {labels[idx]:30s} {score:.4f}")


if __name__ == "__main__":
    main()
