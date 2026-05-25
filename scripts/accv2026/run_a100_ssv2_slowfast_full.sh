#!/usr/bin/env bash
# SlowFast R50 full SSV2 training: 10 epochs, 2xA100 DDP, batch 16/GPU.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

EPOCHS="${EPOCHS:-10}"
CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_slowfast_r50_ssv2_full_e${EPOCHS}_a100}"
OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/slowfast_r50_ssv2_full_e${EPOCHS}_a100}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
SUMMARY="${OUT_DIR}/somethingv2_validation_${CHECKPOINT_NAME}_fixed_budget_summary.csv"
EVAL_MANIFEST="${EVAL_MANIFEST:-evaluations/accv2026/manifests/somethingv2_val_20_per_class.csv}"

mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs

echo "[a100-slowfast-full] GPU status"
nvidia-smi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  echo "[a100-slowfast-full] Training SlowFast R50 full SSV2 — ${EPOCHS} epochs, 2xA100 DDP, batch ${BATCH_SIZE:-16}/GPU"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" scripts/accv2026/train_slowfast.py \
    --ddp \
    --data-root data/Something_data \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --lr "${LR:-1e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.05}" \
    --num-workers "${NUM_WORKERS:-6}" \
    --input-size 224 \
    --save-path "${CHECKPOINT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_RUN_NAME:-train-a100ddp-slowfast-r50-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 slowfast full ddp train "job-${ACCV_JOB_ID}"
else
  echo "[a100-slowfast-full] Checkpoint already exists: ${CHECKPOINT}"
fi

if [[ ! -f "${SUMMARY}" ]]; then
  echo "[a100-slowfast-full] Fixed-budget evaluation on val_20_per_class"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python scripts/accv2026/02_run_fixed_budget_eval.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name somethingv2 \
    --split validation \
    --checkpoint "${CHECKPOINT}" \
    --budgets ${BUDGETS:-4 8 16 32} \
    --model-frames "${MODEL_FRAMES:-32}" \
    --batch-size "${EVAL_BATCH_SIZE:-8}" \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "${WANDB_EVAL_RUN_NAME:-eval-a100-slowfast-r50-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}}" \
    --wandb-tags accv2026 a100 ssv2 slowfast full evaluation eval "job-${ACCV_JOB_ID}"
else
  echo "[a100-slowfast-full] Summary already exists: ${SUMMARY}"
fi

echo "[a100-slowfast-full] Computing temporal metrics"
python scripts/accv2026/05_compute_temporal_metrics.py \
  --summary "${SUMMARY}" \
  --output "${OUT_DIR}/temporal_metrics.csv"

echo "[a100-slowfast-full] Done"
