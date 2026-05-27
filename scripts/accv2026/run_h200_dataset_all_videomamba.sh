#!/usr/bin/env bash
# Run VideoMamba on ALL 7 datasets sequentially (idempotent — skips completed).
# Activates .venv_mamba; do not mix with .venv.
set -uo pipefail   # no -e: one dataset failure doesn't kill the rest

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv_mamba/bin/activate ]]; then
    source .venv_mamba/bin/activate
else
    echo "[ERROR] .venv_mamba not found" >&2
    exit 1
fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export EPOCHS="${EPOCHS:-10}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-16}"

DATASETS="${DATASETS:-ssv2 ucf101 hmdb51 driveact diving48 autsl epic_kitchens}"

echo "========================================================"
echo "VideoMamba — ALL datasets — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
nvidia-smi | head -10 || true

for ds in $DATASETS; do
    echo ""
    echo "──────────────────────────────────────────"
    echo "  START: videomamba on ${ds}"
    echo "──────────────────────────────────────────"
    DATASET="$ds" bash scripts/accv2026/run_h200_multidata_videomamba.sh || \
        echo "  [WARN] videomamba on ${ds} exited non-zero — continuing"
    echo "  [DONE] videomamba on ${ds}"
done

echo ""
echo "========================================================"
echo "VideoMamba ALL datasets done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
