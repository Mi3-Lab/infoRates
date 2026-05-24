#!/usr/bin/env bash
set -euo pipefail

# Simple pipeline: build manifest if missing, then run DDP evaluation using config.yaml
# Requires: PyYAML installed in the venv

CONFIG=${1:-config.yaml}

# Activate venv if not already
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
  fi
fi

# Read values from YAML using Python (no external yq dependency)
read_yaml() {
  python - <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
key = sys.argv[2]
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
# Support nested keys via dot notation
cur = cfg
for part in key.split('.'):
    cur = cur.get(part, {}) if isinstance(cur, dict) else None
print(cur if isinstance(cur, (str, int, float, bool)) or cur is None else '')
PY
}

NUM_GPUS=$(python - <<PY
import yaml, os
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
# Allow NPROC_PER_NODE env var to override config
print(int(os.getenv("NPROC_PER_NODE", cfg.get("num_gpus", 2))))
PY
)

VIDEO_ROOT=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
print(cfg.get("video_root","UCF101_data/UCF-101"))
PY
)

MANIFEST=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
print(cfg.get("eval_manifest","UCF101_data/ucf101_fixedlen_50f.csv"))
PY
)

BUILD_IF_MISSING=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
print(bool(cfg.get("build_manifest_if_missing", True)))
PY
)

FIXED_FRAMES=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
print(int(cfg.get("fixed_frames",50)))
PY
)

FIXED_OUT_DIR=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
print(cfg.get("fixed_out_dir","UCF101_data/UCF101_fixed_50f"))
PY
)

EVAL_RUN_NAME=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
    cfg=yaml.safe_load(f)
print(cfg.get("eval_wandb_run_name","eval-finetuned-timesformer-ddp"))
PY
)

# Per-class output path and sample size from config (defaults applied if missing)
PER_CLASS_OUT=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
  cfg=yaml.safe_load(f)
print(cfg.get("eval_per_class_out","UCF101_data/results/ucf101_50f_per_class.csv"))
PY
)

PER_CLASS_SAMPLE_SIZE=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
  cfg=yaml.safe_load(f)
print(int(cfg.get("eval_per_class_sample_size", -1)))
PY
)

## Build manifest if missing (prefer building from existing clips; do NOT re-split videos)
if [[ ! -f "$MANIFEST" ]] && [[ "$BUILD_IF_MISSING" == "True" || "$BUILD_IF_MISSING" == "true" ]]; then
  echo "Manifest not found: $MANIFEST. Attempting to build using unified CLI (scripts/manage_data.py)..."
  if [[ -d "$FIXED_OUT_DIR" ]]; then
    SPLIT_FILE=$(python - <<PY
import yaml
print(yaml.safe_load(open("$CONFIG")).get("split_file",""))
PY
)
    python scripts/manage_data.py build-manifest --clips-dir "$FIXED_OUT_DIR" --out "$MANIFEST" ${SPLIT_FILE:+--split-file "$SPLIT_FILE"}
  else
    echo "Clips dir $FIXED_OUT_DIR not found. Building clips now using existing script..."
    python scripts/build_clips.py \
      --video-root "$VIDEO_ROOT" \
      --out "$FIXED_OUT_DIR" \
      --frames "$FIXED_FRAMES" \
      --manifest "$MANIFEST"
  fi
fi

# Derive model path and tag for unique outputs per model
MODEL_PATH=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
  cfg=yaml.safe_load(f)
print(cfg.get("save_path","models/timesformer_ucf101_ddp"))
PY
)
MODEL_TAG="$(basename "$MODEL_PATH")"

# Results directory
RESULTS_DIR=$(python - <<PY
import yaml
with open("$CONFIG","r") as f:
  cfg=yaml.safe_load(f)
print(cfg.get("eval_results_dir","UCF101_data/results"))
PY
)
mkdir -p "$RESULTS_DIR"

# Compose unique output paths per model
RESULTS_CSV="$RESULTS_DIR/${MODEL_TAG}_temporal_sampling.csv"
PER_CLASS_OUT="$RESULTS_DIR/${MODEL_TAG}_per_class.csv"

# Launch DDP evaluation via torchrun using config
echo "Starting DDP evaluation with $NUM_GPUS GPUs using $CONFIG..."

# Check GPU availability when NUM_GPUS > 0
GPU_AVAIL=$(python - <<PY
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)
if [[ "$NUM_GPUS" -gt 0 && "$GPU_AVAIL" -eq 0 ]]; then
  echo "Error: No GPUs detected. Please run inside an allocated GPU node (use salloc/srun)." >&2
  exit 1
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/run_eval.py --config "$CONFIG" --ddp \
  --model-path "$MODEL_PATH" \
  --manifest "$MANIFEST" \
  --wandb-run-name "$EVAL_RUN_NAME" \
  --out "$RESULTS_CSV" \
  --per-class \
  --per-class-out "$PER_CLASS_OUT" \
  --per-class-sample-size "$PER_CLASS_SAMPLE_SIZE"

# After evaluation, generate plots and summary automatically
PER_CLASS_CSV="$PER_CLASS_OUT"

if [[ -f "$RESULTS_CSV" ]]; then
  echo "Generating plots and summary from $RESULTS_CSV ..."
  # Log plots to W&B and save locally; include per-class CSV if present
  if [[ -f "$PER_CLASS_CSV" ]]; then
    python scripts/plot_results.py --config "$CONFIG" --csv "$RESULTS_CSV" --per-class-csv "$PER_CLASS_CSV" --wandb
  else
    python scripts/plot_results.py --config "$CONFIG" --csv "$RESULTS_CSV" --wandb
  fi
else
  echo "Warning: Results CSV not found at $RESULTS_CSV; skipping plotting." >&2
fi
