---
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

# wesleymaia/videomamba3-tiny-complex-ucf101-smoke

VideoMamba3 is an experimental VideoMamba backbone using a Mamba-3-style state
space mixer with trapezoidal discretization, data-dependent complex rotations,
and optional low-rank MIMO updates.

## Checkpoint

- Architecture: `videomamba3_tiny`
- Variant: `complex`
- Frames: `2`
- Input size: `112`
- Depth: `24`
- Labels: `101`
- Validation accuracy in source run: `1.0`
- SSM config: `{"d_state": 16, "expand": 1, "headdim": 32, "mimo_rank": 2}`

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
