#!/usr/bin/env bash
set -euo pipefail

# Multi-model DDP Training Launcher
# Runs scripts/train_multimodel.py with torchrun across multiple GPUs.
#
# Usage examples:
#   bash scripts/train_ddp.sh --model videomae --gpus 2 --epochs 5
#   NPROC_PER_NODE=4 bash scripts/train_ddp.sh --model vivit --batch-size 4
#   bash scripts/train_ddp.sh --model all --gpus 2 --no-wandb
#
# Defaults are read from config.yaml but can be overridden by flags.

HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$HERE_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="config.yaml"
MODEL=""
GPUS=""
EPOCHS=""
BATCH_SIZE=""
LR=""
GRAD_ACCUM=""
RUN_NAME=""
NO_WANDB="false"
VIDEO_ROOT=""
SAVE_PATH=""
DRY_RUN="false"

print_usage() {
  cat <<USAGE
Multi-model DDP Training Launcher

Flags:
  --config <path>          Path to config file (default: config.yaml)
  --model <name>           Model to train: timesformer | videomae | vivit | all
  --gpus <N>               Number of GPUs (overrides env NPROC_PER_NODE)
  --epochs <N>             Number of epochs (overrides config)
  --batch-size <N>         Batch size per GPU (overrides config)
  --lr <float>             Learning rate (overrides config)
  --grad-accum <N>         Gradient accumulation steps (overrides config)
  --run-name <str>         WandB run name
  --no-wandb               Disable Weights & Biases logging
  --video-root <path>      UCF101 root (overrides config)
  --save-path <path>       Model save directory (overrides config)
  --dry-run                Print the torchrun command without executing
  -h | --help              Show this help

Environment:
  NPROC_PER_NODE           GPU count if --gpus not provided

Examples:
  bash scripts/train_ddp.sh --model videomae --gpus 2 --epochs 5
  bash scripts/train_ddp.sh --model all --gpus 4 --grad-accum 2 --no-wandb
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --no-wandb) NO_WANDB="true"; shift 1;;
    --video-root) VIDEO_ROOT="$2"; shift 2;;
    --save-path) SAVE_PATH="$2"; shift 2;;
    --dry-run) DRY_RUN="true"; shift 1;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; print_usage; exit 1;;
  esac
done

# Activate venv if available
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Check torchrun availability
if ! command -v torchrun >/dev/null 2>&1; then
  echo "Error: torchrun not found in PATH. Activate your environment or install PyTorch." >&2
  exit 1
fi

# Helper: read YAML key via Python (no yq dependency)
py_yaml() {
  local key=$1
  python - "$CONFIG" "$key" <<'PY'
import sys, yaml
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f) or {}
def read(cfg, dotted, default=None):
    cur = cfg
    for p in dotted.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur
val = read(cfg, key, '')
if isinstance(val, (str, int, float, bool)):
    print(val)
else:
    print('')
PY
}

# Defaults from config.yaml
CFG_MODEL=${MODEL:-$(py_yaml model_name)}
CFG_EPOCHS=${EPOCHS:-$(py_yaml train_epochs)}
CFG_BATCH=${BATCH_SIZE:-$(py_yaml train_batch_size)}
CFG_LR=${LR:-$(py_yaml train_learning_rate)}
CFG_GRAD=${GRAD_ACCUM:-$(py_yaml train_gradient_accumulation_steps)}
CFG_VIDEO_ROOT=${VIDEO_ROOT:-$(py_yaml train_video_root)}
CFG_SAVE_PATH=${SAVE_PATH:-$(py_yaml train_save_path)}
CFG_WANDB_PROJ=$(py_yaml train_wandb_project)
CFG_DISABLE_WANDB=$(py_yaml train_disable_wandb)

# GPU count: prefer --gpus, then env NPROC_PER_NODE, then config, finally detected GPUs
DETECTED_GPUS=$(python - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
CFG_GPUS=${GPUS:-${NPROC_PER_NODE:-$(py_yaml train_num_gpus)}}
CFG_GPUS=${CFG_GPUS:-$DETECTED_GPUS}
CFG_GPUS=${CFG_GPUS:-1}

# Basic validations
if [[ -z "$CFG_MODEL" ]]; then
  echo "Error: model not specified and model_name missing in $CONFIG" >&2
  exit 1
fi
case "$CFG_MODEL" in
  timesformer|videomae|vivit|all) : ;; 
  *) echo "Error: invalid --model '$CFG_MODEL' (use timesformer|videomae|vivit|all)" >&2; exit 1;;
esac

if [[ "$DRY_RUN" != "true" && "$CFG_GPUS" -gt 0 ]]; then
  AVAIL=$(python - <<'PY'
try:
    import torch
    print(1 if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)
  if [[ "$AVAIL" -eq 0 ]]; then
    echo "Error: No GPUs available. Please run on a GPU node." >&2
    exit 1
  fi
fi

WANDB_FLAG=()
if [[ "$NO_WANDB" == "true" || "$CFG_DISABLE_WANDB" == "True" || "$CFG_DISABLE_WANDB" == "true" ]]; then
  WANDB_FLAG+=(--no-wandb)
fi
if [[ -n "${CFG_WANDB_PROJ:-}" ]]; then
  WANDB_FLAG+=(--wandb-project "$CFG_WANDB_PROJ")
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  WANDB_FLAG+=(--wandb-run-name "$RUN_NAME")
fi

OVERRIDES=(
  --config "$CONFIG"
  --model "$CFG_MODEL"
  --ddp
)
[[ -n "$CFG_EPOCHS" ]] && OVERRIDES+=(--epochs "$CFG_EPOCHS")
[[ -n "$CFG_BATCH" ]] && OVERRIDES+=(--batch-size "$CFG_BATCH")
[[ -n "$CFG_LR" ]] && OVERRIDES+=(--lr "$CFG_LR")
[[ -n "$CFG_GRAD" ]] && OVERRIDES+=(--gradient-accumulation-steps "$CFG_GRAD")
[[ -n "$CFG_VIDEO_ROOT" ]] && OVERRIDES+=(--video-root "$CFG_VIDEO_ROOT")
[[ -n "$CFG_SAVE_PATH" ]] && OVERRIDES+=(--save-path "$CFG_SAVE_PATH")

set -x
CMD=(torchrun --standalone --nproc_per_node="$CFG_GPUS" scripts/train_multimodel.py "${OVERRIDES[@]}" "${WANDB_FLAG[@]}")
set +x

echo "Launching: ${CMD[*]}"
if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
