#!/usr/bin/env bash
# Generic VideoMAE runner for UCF101 / HMDB51 / Diving48.
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
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_videomae_${DATASET}_full_e${EPOCHS}_h200}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/videomae_${DATASET}_full_e${EPOCHS}_h200}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
MANIFEST_DIR="evaluations/accv2026/manifests"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST_DIR}/${DATASET}_val_20_per_class.csv}"
SUMMARY="${OUT_DIR}/${DATASET}_val_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs "${MANIFEST_DIR}"

echo "[multidata-videomae] GPU status"
nvidia-smi

if [[ ! -f "${EVAL_MANIFEST}" ]]; then
  echo "[multidata-videomae] Building eval manifest for ${DATASET}"
  python - <<PYEOF
import sys; sys.path.insert(0, "src")
from info_rates.data.datasets import build_eval_manifest
df = build_eval_manifest("${DATASET}", samples_per_class=20)
df.to_csv("${EVAL_MANIFEST}", index=False)
print(f"  Wrote {len(df)} rows to ${EVAL_MANIFEST}")
PYEOF
fi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[multidata-videomae] Training VideoMAE on ${DATASET} — ${EPOCHS} epochs"
  python scripts/accv2026/train_transformers.py \
    --dataset "${DATASET}" \
    --model videomae \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --lr "${LR:-2e-5}" \
    --weight-decay "${WEIGHT_DECAY:-0.05}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --save-path "${CHECKPOINT}" \
    ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-videomae-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 "${DATASET}" videomae transformer full train "job-${ACCV_JOB_ID}"
else
  echo "[multidata-videomae] Checkpoint exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[multidata-videomae] Fixed-budget evaluation"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name "${DATASET}" \
    --split "val" \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames 16 \
    --batch-size "${EVAL_BATCH_SIZE:-16}" \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-videomae-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 h200 "${DATASET}" videomae transformer full evaluation "job-${ACCV_JOB_ID}"
fi

echo "[multidata-videomae] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[multidata-videomae] Done — VideoMAE on ${DATASET}"
