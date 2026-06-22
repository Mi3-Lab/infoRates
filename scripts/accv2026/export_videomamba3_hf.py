#!/usr/bin/env python3
"""Export a VideoMamba3 checkpoint as a Hugging Face model repository folder."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VM3 = ROOT / "experiments" / "videomamba3"
for path in (ROOT, SRC, VM3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_videomamba3 import load_videomamba3_checkpoint  # noqa: E402


def read_history(checkpoint: Path) -> dict:
    history = checkpoint.with_name(checkpoint.name + "_history.csv")
    if not history.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(history)
        if df.empty:
            return {}
        row = df.iloc[df["val_accuracy"].idxmax()].to_dict()
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()}
    except Exception:
        return {"history_csv": str(history)}


def write_model_card(out_dir: Path, meta: dict, metrics: dict, repo_id: str | None) -> None:
    variant = meta.get("mamba3_variant", "complex")
    model_name = meta.get("model_name", "videomamba3")
    dataset = meta.get("dataset", "video-classification")
    val_acc = meta.get("val_acc", metrics.get("val_accuracy", "n/a"))
    card = f"""---
library_name: pytorch
pipeline_tag: video-classification
tags:
- videomamba3
- mamba-3
- state-space-models
- video-classification
- pytorch
license: mit
---

# {repo_id or model_name}

VideoMamba3 is an experimental VideoMamba backbone using a Mamba-3-style state
space mixer with trapezoidal discretization, data-dependent complex rotations,
and optional low-rank MIMO updates.

## Checkpoint

- Architecture: `{model_name}`
- Variant: `{variant}`
- Frames: `{meta.get("num_frames")}`
- Input size: `{meta.get("input_size")}`
- Depth: `{meta.get("depth")}`
- Labels: `{meta.get("num_labels")}`
- Validation accuracy in source run: `{val_acc}`
- SSM config: `{json.dumps(meta.get("ssm_cfg", {}), sort_keys=True)}`

## Caveat

This release currently uses a pure-PyTorch reference scan. It is intended for
research, reproducibility, and community iteration. Full-resolution training
will benefit from a fused/chunked scan kernel.

## Loading

```python
import json
import sys
import torch

sys.path.insert(0, ".")
from videomamba3 import VisionMamba

meta = json.load(open("accv_meta.json"))
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
state = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

## Files

- `pytorch_model.bin`: model weights
- `accv_meta.json`: training/config metadata
- `videomamba3.py`, `mamba3_core.py`: model code
- `processor_config.json`: ImageNet normalization and video preprocessing settings
"""
    (out_dir / "README.md").write_text(card)


def export_checkpoint(args) -> Path:
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, meta = load_videomamba3_checkpoint(checkpoint, device="cpu")
    raw = model.backbone
    torch.save(raw.state_dict(), out_dir / "pytorch_model.bin")

    meta = dict(meta)
    meta.setdefault("depth", raw.get_num_layers() if hasattr(raw, "get_num_layers") else len(raw.layers))
    meta["export_format"] = "huggingface-folder"
    meta["source_checkpoint"] = str(checkpoint)
    (out_dir / "accv_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "architectures": ["VisionMamba"],
        "model_type": "videomamba3",
        "backend": "videomamba3",
        "model_name": meta.get("model_name"),
        "mamba3_variant": meta.get("mamba3_variant"),
        "num_frames": meta.get("num_frames"),
        "input_size": meta.get("input_size"),
        "depth": meta.get("depth"),
        "embed_dim": meta.get("embed_dim"),
        "num_labels": meta.get("num_labels"),
        "id2label": {str(i): name for i, name in enumerate(meta.get("class_names", []))},
        "label2id": {name: i for i, name in enumerate(meta.get("class_names", []))},
        "ssm_cfg": meta.get("ssm_cfg", {}),
    }, indent=2))
    (out_dir / "processor_config.json").write_text(json.dumps({
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size": meta.get("input_size"),
        "num_frames": meta.get("num_frames"),
        "data_format": "B C T H W",
    }, indent=2))
    (out_dir / "requirements.txt").write_text("torch\ntimm\neinops\nopencv-python\nav\n")
    for filename in ("videomamba3.py", "mamba3_core.py"):
        shutil.copy2(VM3 / filename, out_dir / filename)
    metrics = read_history(checkpoint)
    write_model_card(out_dir, meta, metrics, args.repo_id)
    return out_dir


def maybe_upload(out_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(out_dir))
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="fine_tuned_models/... checkpoint directory")
    p.add_argument("--output-dir", required=True, help="local Hugging Face repository folder")
    p.add_argument("--repo-id", default=None, help="optional Hugging Face repo id, e.g. user/videomamba3-tiny")
    p.add_argument("--push", action="store_true", help="upload the exported folder to Hugging Face Hub")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = export_checkpoint(args)
    print(f"Exported VideoMamba3 checkpoint to {out_dir}")
    if args.push:
        if not args.repo_id:
            raise ValueError("--push requires --repo-id")
        maybe_upload(out_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
