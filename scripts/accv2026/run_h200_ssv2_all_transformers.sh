#!/usr/bin/env bash
# Treina TimeSformer + VideoMAE em paralelo (GPU 0 e GPU 1), depois ViViT.
# Requer node com 2x H200. Ideal para job sbatch com --gres=gpu:nvidia_h200_nvl:2
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/infoRates/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export ACCV_JOB_ID="${ACCV_JOB_ID:-${SLURM_JOB_ID:-manual}}"

EPOCHS="${EPOCHS:-10}"
LOG=evaluations/accv2026/logs
mkdir -p "${HF_HOME}" evaluations/accv2026/fixed_budget fine_tuned_models "${LOG}"

echo "[h200-all] Iniciando em $(hostname) — $(date)"
nvidia-smi

# ── Fase 1: TimeSformer (GPU 0) + VideoMAE (GPU 1) em paralelo ──────────────

TSF_CKPT="${TSF_CHECKPOINT:-fine_tuned_models/accv2026_timesformer_ssv2_full_e${EPOCHS}_h200}"
VME_CKPT="${VME_CHECKPOINT:-fine_tuned_models/accv2026_videomae_ssv2_full_e${EPOCHS}_h200}"

if [[ ! -f "${TSF_CKPT}/accv_meta.json" ]]; then
  echo "[h200-all] GPU 0 → TimeSformer ${EPOCHS} épocas"
  CUDA_VISIBLE_DEVICES=0 python scripts/accv2026/train_transformers.py \
    --data-root data/Something_data \
    --model timesformer \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --lr "${LR:-2e-5}" \
    --weight-decay 0.05 \
    --num-workers "${NUM_WORKERS:-12}" \
    --save-path "${TSF_CKPT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "train-h200-timesformer-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}" \
    --wandb-tags accv2026 h200 ssv2 timesformer full train "job-${ACCV_JOB_ID}" \
    2>&1 | tee "${LOG}/timesformer-h200-${ACCV_JOB_ID}.out" &
  PID_TSF=$!
else
  echo "[h200-all] TimeSformer checkpoint já existe: ${TSF_CKPT}"
  PID_TSF=""
fi

if [[ ! -f "${VME_CKPT}/accv_meta.json" ]]; then
  echo "[h200-all] GPU 1 → VideoMAE ${EPOCHS} épocas"
  CUDA_VISIBLE_DEVICES=1 python scripts/accv2026/train_transformers.py \
    --data-root data/Something_data \
    --model videomae \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --lr "${LR:-2e-5}" \
    --weight-decay 0.05 \
    --num-workers "${NUM_WORKERS:-12}" \
    --save-path "${VME_CKPT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "train-h200-videomae-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}" \
    --wandb-tags accv2026 h200 ssv2 videomae full train "job-${ACCV_JOB_ID}" \
    2>&1 | tee "${LOG}/videomae-h200-${ACCV_JOB_ID}.out" &
  PID_VME=$!
else
  echo "[h200-all] VideoMAE checkpoint já existe: ${VME_CKPT}"
  PID_VME=""
fi

# Aguarda os dois terminarem
[[ -n "${PID_TSF:-}" ]] && wait "${PID_TSF}" && echo "[h200-all] TimeSformer concluído — $(date)"
[[ -n "${PID_VME:-}" ]] && wait "${PID_VME}" && echo "[h200-all] VideoMAE concluído — $(date)"

# ── Fase 2: ViViT (GPU 0) ────────────────────────────────────────────────────

VVT_CKPT="${VVT_CHECKPOINT:-fine_tuned_models/accv2026_vivit_ssv2_full_e${EPOCHS}_h200}"

if [[ ! -f "${VVT_CKPT}/accv_meta.json" ]]; then
  echo "[h200-all] GPU 0 → ViViT ${EPOCHS} épocas"
  CUDA_VISIBLE_DEVICES=0 python scripts/accv2026/train_transformers.py \
    --data-root data/Something_data \
    --model vivit \
    --epochs "${EPOCHS}" \
    --batch-size "${VIVIT_BATCH_SIZE:-32}" \
    --lr "${VIVIT_LR:-1e-5}" \
    --weight-decay 0.05 \
    --num-workers "${NUM_WORKERS:-12}" \
    --save-path "${VVT_CKPT}" \
    --wandb-project "${WANDB_PROJECT:-inforates-accv2026}" \
    --wandb-run-name "train-h200-vivit-ssv2-full-e${EPOCHS}-job${ACCV_JOB_ID}" \
    --wandb-tags accv2026 h200 ssv2 vivit full train "job-${ACCV_JOB_ID}" \
    2>&1 | tee "${LOG}/vivit-h200-${ACCV_JOB_ID}.out"
else
  echo "[h200-all] ViViT checkpoint já existe: ${VVT_CKPT}"
fi

echo "[h200-all] Todos os transformers concluídos — $(date)"
