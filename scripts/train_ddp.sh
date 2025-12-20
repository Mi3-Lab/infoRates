#!/bin/bash
# DDP Training Script for 2 GPUs

# Usage: bash scripts/train_ddp.sh

# Prefer the project virtualenv to ensure dependencies like wandb are available
VENV_BIN="$(dirname "$0")/../.venv/bin"
TORCHRUN_BIN="$VENV_BIN/torchrun"

if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "torchrun not found at $TORCHRUN_BIN; did you create the venv and install deps?" >&2
  exit 1
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-2}

"$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_timesformer.py \
  --config config.yaml
