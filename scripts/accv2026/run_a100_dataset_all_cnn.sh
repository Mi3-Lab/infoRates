#!/usr/bin/env bash
# Run ALL CNN models (r3d_18, mc3_18, slowfast_r50) on a single dataset sequentially.
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
echo "ALL CNN models on ${DATASET} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
nvidia-smi | head -10

run_model() {
    local model=$1
    echo ""
    echo "──────────────────────────────────────────"
    echo "  START: ${model} on ${DATASET}"
    echo "──────────────────────────────────────────"
    if [[ "$model" == "slowfast_r50" ]]; then
        BATCH_SIZE="${BATCH_SIZE_SF:-8}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_SF:-8}" \
            RESUME_FROM="${RESUME_FROM_SF:-}" \
            bash scripts/accv2026/run_a100_multidata_slowfast.sh
    else
        local resume_var="RESUME_FROM_${model^^}"  # e.g. RESUME_FROM_R3D_18 → too long; use model slug
        local resume_slug="RESUME_FROM_$(echo "$model" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
        local resume_val="${!resume_slug:-}"
        MODEL_NAME="$model" BATCH_SIZE="${BATCH_SIZE_CNN:-32}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_CNN:-24}" \
            RESUME_FROM="$resume_val" \
            bash scripts/accv2026/run_a100_multidata_torchvision.sh
    fi
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  [WARN] ${model} exited with code ${rc} — continuing"
    else
        echo "  [OK] ${model} done"
    fi
}

for model in r3d_18 mc3_18 slowfast_r50 r2plus1d_18; do
    run_model "$model" || true
done

echo ""
echo "========================================================"
echo "ALL CNN done on ${DATASET} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
