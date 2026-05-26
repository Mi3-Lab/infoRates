#!/usr/bin/env bash
# Generic TorchVision 3D CNN runner for UCF101 / HMDB51 / Diving48.
# Required env vars: DATASET, MODEL_NAME
# Optional overrides: DATA_ROOT, EPOCHS, BATCH_SIZE, NUM_WORKERS, NPROC_PER_NODE
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

DATASET="${DATASET:?DATASET must be set (ucf101|hmdb51|diving48)}"
MODEL_NAME="${MODEL_NAME:-r2plus1d_18}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL_NAME//_/-}}"
EPOCHS="${EPOCHS:-10}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_${MODEL_NAME}_${DATASET}_full_e${EPOCHS}_a100}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/${MODEL_NAME}_${DATASET}_full_e${EPOCHS}_a100}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
MANIFEST_DIR="evaluations/accv2026/manifests"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST_DIR}/${DATASET}_val_20_per_class.csv}"
SUMMARY="${OUT_DIR}/${DATASET}_val_${CHECKPOINT_NAME}_fixed_budget_summary.csv"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs "${MANIFEST_DIR}"

echo "[multidata-torchvision] GPU status"
nvidia-smi

# Build eval manifest if missing
if [[ ! -f "${EVAL_MANIFEST}" ]]; then
  echo "[multidata-torchvision] Building eval manifest for ${DATASET}"
  python - <<PYEOF
import sys; sys.path.insert(0, "src")
from info_rates.data.datasets import build_eval_manifest
df = build_eval_manifest("${DATASET}", samples_per_class=20)
df.to_csv("${EVAL_MANIFEST}", index=False)
print(f"  Wrote {len(df)} rows to ${EVAL_MANIFEST}")
PYEOF
fi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[multidata-torchvision] Training ${MODEL_NAME} on ${DATASET} — ${EPOCHS} epochs"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" scripts/accv2026/train_torchvision.py \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --lr "${LR:-1e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.05}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --num-frames "${MODEL_FRAMES:-16}" \
    --input-size "${INPUT_SIZE:-112}" \
    --save-path "${CHECKPOINT}" \
    ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-${MODEL_SLUG}-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 "${DATASET}" "${MODEL_NAME}" 3d-cnn full train "job-${ACCV_JOB_ID}"
else
  echo "[multidata-torchvision] Checkpoint exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[multidata-torchvision] Fixed-budget evaluation"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/eval_fixed_budget.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name "${DATASET}" \
    --split "val" \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames "${MODEL_FRAMES:-16}" \
    --batch-size "${EVAL_BATCH_SIZE:-24}" \
    --resize "${INPUT_SIZE:-112}" \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-${MODEL_SLUG}-${DATASET}-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 "${DATASET}" "${MODEL_NAME}" 3d-cnn full evaluation "job-${ACCV_JOB_ID}"
fi

echo "[multidata-torchvision] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[multidata-torchvision] Done — ${MODEL_NAME} on ${DATASET}"
