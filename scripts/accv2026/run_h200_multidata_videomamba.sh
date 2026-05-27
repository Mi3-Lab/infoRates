#!/usr/bin/env bash
# Train VideoMamba on any dataset, then run fixed-budget evaluation.
# Requires .venv_mamba (mamba-ssm + causal-conv1d compiled).
# Required env: DATASET
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

# VideoMamba needs its own venv with mamba-ssm
if [[ -f .venv_mamba/bin/activate ]]; then
    source .venv_mamba/bin/activate
else
    echo "[ERROR] .venv_mamba not found — run slurm_build_mamba_ssm.sbatch first" >&2
    exit 1
fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

DATASET="${DATASET:?DATASET must be set}"
EPOCHS="${EPOCHS:-10}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_videomamba_${DATASET}_full_e${EPOCHS}_h200}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/videomamba_${DATASET}_full_e${EPOCHS}_h200}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
MANIFEST_DIR="evaluations/accv2026/manifests"
# ssv2 uses 'somethingv2' as the manifest prefix
MANIFEST_PREFIX="${DATASET}"
[[ "${DATASET}" == "ssv2" ]] && MANIFEST_PREFIX="somethingv2"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST_DIR}/${MANIFEST_PREFIX}_val_20_per_class.csv}"
SUMMARY="${OUT_DIR}/${DATASET}_val_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

DEFAULT_BATCH="${BATCH_SIZE:-16}"
DEFAULT_EVAL_BATCH="${EVAL_BATCH_SIZE:-16}"
MODEL_FRAMES="${MODEL_FRAMES:-8}"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs "${MANIFEST_DIR}"

echo "[videomamba] GPU: videomamba on ${DATASET}"
nvidia-smi | head -10 || true

# Build eval manifest if missing
if [[ ! -f "${EVAL_MANIFEST}" ]]; then
    echo "[videomamba] Building eval manifest for ${DATASET}"
    python - <<PYEOF
import sys; sys.path.insert(0, "src")
from info_rates.data.datasets import build_eval_manifest
df = build_eval_manifest("${DATASET}", samples_per_class=20)
df.to_csv("${EVAL_MANIFEST}", index=False)
print(f"  Wrote {len(df)} rows to ${EVAL_MANIFEST}")
PYEOF
fi

# Train (skip if checkpoint already exists)
if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
    echo "[videomamba] Training VideoMamba on ${DATASET} — ${EPOCHS} epochs"
    python scripts/accv2026/train_videomamba.py \
        --dataset "${DATASET}" \
        --epochs "${EPOCHS}" \
        --batch-size "${DEFAULT_BATCH}" \
        --lr "${LR:-2e-5}" \
        --weight-decay "${WEIGHT_DECAY:-0.05}" \
        --num-workers "${NUM_WORKERS:-4}" \
        --save-path "${CHECKPOINT}" \
        ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
        --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
        --wandb-run-name "${WANDB_RUN_NAME:-train-videomamba-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
        --wandb-tags accv2026 h200 "${DATASET}" videomamba ssm full train "job-${ACCV_JOB_ID}"
else
    echo "[videomamba] Checkpoint exists: ${CHECKPOINT}"
fi

# Fixed-budget evaluation (skip if summary already exists)
if [[ ! -f "${SUMMARY}" ]]; then
    echo "[videomamba] Fixed-budget evaluation"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/eval_fixed_budget.py \
        --manifest "${EVAL_MANIFEST}" \
        --dataset-name "${DATASET}" \
        --split "val" \
        --checkpoint "${CHECKPOINT}" \
        --budgets ${BUDGETS:-4 8 16 32} \
        --model-frames "${MODEL_FRAMES}" \
        --batch-size "${DEFAULT_EVAL_BATCH}" \
        --resize 224 \
        --output-dir "${OUT_DIR}" \
        --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
        --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-videomamba-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
        --wandb-tags accv2026 h200 "${DATASET}" videomamba ssm full evaluation "job-${ACCV_JOB_ID}"
fi

echo "[videomamba] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
    --summary "${SUMMARY}" \
    --output "${OUT_DIR}/temporal_metrics.csv"

echo "[videomamba] Done — VideoMamba on ${DATASET}"
