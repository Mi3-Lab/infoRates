# VideoMamba3 Release Guide

This document tracks the path from an experimental checkpoint to a community
release on Hugging Face.

## What Is Ready

- `experiments/videomamba3/mamba3_core.py`: pure-PyTorch Mamba-3-style SSM core.
- `experiments/videomamba3/videomamba3.py`: VideoMamba backbone with `BiMamba3`.
- `scripts/accv2026/train_videomamba3.py`: UCF-101 training with full metadata.
- `scripts/accv2026/export_videomamba3_hf.py`: Hugging Face folder export and upload.
- `examples/videomamba3_inference.py`: minimal video classification inference.
- `docs/VIDEOMAMBA3_PAPER_DRAFT.md`: paper draft and experiment status.

## Recommended Public Checkpoint

Use a small reference checkpoint first:

```bash
DEPTH=8 \
NUM_FRAMES=2 \
INPUT_SIZE=112 \
MAX_TRAIN_SAMPLES=512 \
MAX_VAL_SAMPLES=128 \
bash scripts/accv2026/submit_videomamba3_ablation_h200.sh
```

This is intentionally modest. The current recurrence is a clear PyTorch
reference implementation, not yet a fused production kernel.

## Export

```bash
python scripts/accv2026/export_videomamba3_hf.py \
  --checkpoint fine_tuned_models/<checkpoint-dir> \
  --output-dir evaluations/accv2026/videomamba3/hf_export \
  --repo-id <user-or-org>/<repo-name>
```

The exported folder contains:

- `pytorch_model.bin`
- `config.json`
- `accv_meta.json`
- `processor_config.json`
- `README.md`
- `videomamba3.py`
- `mamba3_core.py`
- `requirements.txt`

## Smoke Inference

```bash
python examples/videomamba3_inference.py \
  --model-dir evaluations/accv2026/videomamba3/hf_export \
  --video path/to/video.mp4 \
  --top-k 5
```

## Push To Hugging Face

Log in once:

```bash
huggingface-cli login
```

Then push:

```bash
python scripts/accv2026/export_videomamba3_hf.py \
  --checkpoint fine_tuned_models/<checkpoint-dir> \
  --output-dir evaluations/accv2026/videomamba3/hf_export \
  --repo-id <user-or-org>/<repo-name> \
  --push
```

## Release Notes To Be Honest About

- The current model is a research/reference implementation.
- Full-depth `tiny` has 24 blocks and is slow with the explicit scan.
- Public demo checkpoints should start at `--depth 8`.
- A fused or chunked scan is the next major engineering task before large-scale
  training at `224x224` and long frame budgets.
