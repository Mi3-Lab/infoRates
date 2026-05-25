#!/usr/bin/env bash
# Generic SlowFast R50 runner for UCF101 / HMDB51 / Diving48.
# Required env vars: DATASET
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

DATASET="${DATASET:?DATASET must be set (ucf101|hmdb51|diving48)}"
EPOCHS="${EPOCHS:-10}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_slowfast_r50_${DATASET}_full_e${EPOCHS}_a100}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/slowfast_r50_${DATASET}_full_e${EPOCHS}_a100}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
MANIFEST_DIR="evaluations/accv2026/manifests"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST_DIR}/${DATASET}_val_20_per_class.csv}"
SUMMARY="${OUT_DIR}/${DATASET}_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs "${MANIFEST_DIR}"

echo "[multidata-slowfast] GPU status"
nvidia-smi

if [[ ! -f "${EVAL_MANIFEST}" ]]; then
  echo "[multidata-slowfast] Building eval manifest for ${DATASET}"
  python - <<PYEOF
import sys; sys.path.insert(0, "src")
from info_rates.data.datasets import build_eval_manifest
df = build_eval_manifest("${DATASET}", samples_per_class=20)
df.to_csv("${EVAL_MANIFEST}", index=False)
print(f"  Wrote {len(df)} rows to ${EVAL_MANIFEST}")
PYEOF
fi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[multidata-slowfast] Training SlowFast R50 on ${DATASET} — ${EPOCHS} epochs"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" scripts/accv2026/train_slowfast.py \
    --dataset "${DATASET}" \
    --ddp \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --lr "${LR:-1e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.05}" \
    --num-workers "${NUM_WORKERS:-2}" \
    --input-size 224 \
    --save-path "${CHECKPOINT}" \
    ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-slowfast-r50-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 "${DATASET}" slowfast full train "job-${ACCV_JOB_ID}"
else
  echo "[multidata-slowfast] Checkpoint exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[multidata-slowfast] Fixed-budget evaluation"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name "${DATASET}" \
    --split "val" \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames 32 \
    --batch-size "${EVAL_BATCH_SIZE:-8}" \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-slowfast-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 "${DATASET}" slowfast full evaluation "job-${ACCV_JOB_ID}"
fi

echo "[multidata-slowfast] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[multidata-slowfast] Done — SlowFast R50 on ${DATASET}"
