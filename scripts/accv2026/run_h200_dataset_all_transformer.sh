#!/usr/bin/env bash
# Run ALL transformer models (timesformer, vivit) on a single dataset sequentially.
# Skips any model whose checkpoint/summary already exists (idempotent).
# Required env: DATASET
set -uo pipefail   # no -e: one model failure doesn't kill the rest

cd /data/wesleyferreiramaia/infoRates
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export EPOCHS="${EPOCHS:-10}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

DATASET="${DATASET:?DATASET must be set}"
echo "========================================================"
echo "ALL transformer models on ${DATASET} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
nvidia-smi | head -10

run_model() {
    local model=$1
    echo ""
    echo "──────────────────────────────────────────"
    echo "  START: ${model} on ${DATASET}"
    echo "──────────────────────────────────────────"
    MODEL="$model" bash scripts/accv2026/run_h200_multidata_transformer.sh
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  [WARN] ${model} exited with code ${rc} — continuing"
    else
        echo "  [OK] ${model} done"
    fi
}

for model in timesformer vivit; do
    run_model "$model" || true
done

echo ""
echo "========================================================"
echo "ALL transformers done on ${DATASET} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
