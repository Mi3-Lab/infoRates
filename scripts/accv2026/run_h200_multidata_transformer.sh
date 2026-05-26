#!/usr/bin/env bash
# Generic transformer runner for TimeSformer / ViViT on any dataset.
# Required env vars: DATASET, MODEL  (e.g. MODEL=timesformer or MODEL=vivit)
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

DATASET="${DATASET:?DATASET must be set}"
MODEL="${MODEL:?MODEL must be set (timesformer|vivit)}"
EPOCHS="${EPOCHS:-10}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_${MODEL}_${DATASET}_full_e${EPOCHS}_h200}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/${MODEL}_${DATASET}_full_e${EPOCHS}_h200}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
MANIFEST_DIR="evaluations/accv2026/manifests"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST_DIR}/${DATASET}_val_20_per_class.csv}"
SUMMARY="${OUT_DIR}/${DATASET}_val_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

# ViViT uses 32 frames; TimeSformer uses 8
if [[ "$MODEL" == "vivit" ]]; then
    MODEL_FRAMES="${MODEL_FRAMES:-32}"
    DEFAULT_BATCH="${BATCH_SIZE:-8}"
    DEFAULT_EVAL_BATCH="${EVAL_BATCH_SIZE:-8}"
else
    MODEL_FRAMES="${MODEL_FRAMES:-8}"
    DEFAULT_BATCH="${BATCH_SIZE:-16}"
    DEFAULT_EVAL_BATCH="${EVAL_BATCH_SIZE:-16}"
fi

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs "${MANIFEST_DIR}"

echo "[multidata-transformer] GPU: ${MODEL} on ${DATASET}"
nvidia-smi

if [[ ! -f "${EVAL_MANIFEST}" ]]; then
    echo "[multidata-transformer] Building eval manifest for ${DATASET}"
    python - <<PYEOF
import sys; sys.path.insert(0, "src")
from info_rates.data.datasets import build_eval_manifest
df = build_eval_manifest("${DATASET}", samples_per_class=20)
df.to_csv("${EVAL_MANIFEST}", index=False)
print(f"  Wrote {len(df)} rows to ${EVAL_MANIFEST}")
PYEOF
fi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
    echo "[multidata-transformer] Training ${MODEL} on ${DATASET} — ${EPOCHS} epochs"
    python scripts/accv2026/train_transformers.py \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --epochs "${EPOCHS}" \
        --batch-size "${DEFAULT_BATCH}" \
        --lr "${LR:-2e-5}" \
        --weight-decay "${WEIGHT_DECAY:-0.05}" \
        --num-workers "${NUM_WORKERS:-4}" \
        --save-path "${CHECKPOINT}" \
        ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
        --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
        --wandb-run-name "${WANDB_RUN_NAME:-train-${MODEL}-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
        --wandb-tags accv2026 h200 "${DATASET}" "${MODEL}" transformer full train "job-${ACCV_JOB_ID}"
else
    echo "[multidata-transformer] Checkpoint exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
    echo "[multidata-transformer] Fixed-budget evaluation"
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
        --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-${MODEL}-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
        --wandb-tags accv2026 h200 "${DATASET}" "${MODEL}" transformer full evaluation "job-${ACCV_JOB_ID}"
fi

echo "[multidata-transformer] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
    --summary "${SUMMARY}" \
    --output "${OUT_DIR}/temporal_metrics.csv"

echo "[multidata-transformer] Done — ${MODEL} on ${DATASET}"
