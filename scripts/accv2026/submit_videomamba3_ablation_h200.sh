#!/usr/bin/env bash
# Submit VideoMamba3 ablations on H200.
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

DATASET="${DATASET:-ucf101}"
MODEL_SIZE="${MODEL_SIZE:-tiny}"
DEPTH="${DEPTH:-8}"
VARIANTS="${VARIANTS:-trapezoidal complex mimo}"
EPOCHS="${EPOCHS:-1}"
NUM_FRAMES="${NUM_FRAMES:-2}"
INPUT_SIZE="${INPUT_SIZE:-112}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SSM_D_STATE="${SSM_D_STATE:-16}"
SSM_EXPAND="${SSM_EXPAND:-1}"
SSM_HEADDIM="${SSM_HEADDIM:-32}"
SSM_MIMO_RANK="${SSM_MIMO_RANK:-2}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-128}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

for variant in ${VARIANTS}; do
  save_path="fine_tuned_models/accv2026_videomamba3_${MODEL_SIZE}_${variant}_${DATASET}_${INPUT_SIZE}r_f${NUM_FRAMES}_e${EPOCHS}_h200"
  job_name="vm3-${variant}-${DATASET}"
  echo "[submit] ${job_name} -> ${save_path}"
  sbatch \
    --partition=cenvalarc.gpu \
    --gres=gpu:nvidia_h200_nvl:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time="${TIME_LIMIT}" \
    --job-name="${job_name}" \
    --output="evaluations/accv2026/logs/${job_name}-%j.out" \
    --error="evaluations/accv2026/logs/${job_name}-%j.err" \
    --wrap="cd /data/wesleyferreiramaia/infoRates && \
      source .venv_mamba/bin/activate && \
      export PYTHONPATH=src && \
      export WANDB_MODE=online && \
      export HF_HOME=/scratch/wesleyferreiramaia/hf_unified && \
      python scripts/accv2026/train_videomamba3.py \
        --dataset ${DATASET} \
        --model-size ${MODEL_SIZE} \
        --depth ${DEPTH} \
        --mamba3-variant ${variant} \
        --num-frames ${NUM_FRAMES} \
        --input-size ${INPUT_SIZE} \
        --ssm-d-state ${SSM_D_STATE} \
        --ssm-expand ${SSM_EXPAND} \
        --ssm-headdim ${SSM_HEADDIM} \
        --ssm-mimo-rank ${SSM_MIMO_RANK} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr 2e-5 \
        --weight-decay 0.05 \
        --num-workers 4 \
        --max-train-samples ${MAX_TRAIN_SAMPLES} \
        --max-val-samples ${MAX_VAL_SAMPLES} \
        --no-pretrained \
        --save-path ${save_path} \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-run-name videomamba3-${MODEL_SIZE}-${variant}-${DATASET}-${INPUT_SIZE}r-f${NUM_FRAMES}-e${EPOCHS}-j\${SLURM_JOB_ID} \
        --wandb-tags accv2026 h200 ${DATASET} videomamba3 ${variant} ${MODEL_SIZE} ablation official-mamba3"
done
