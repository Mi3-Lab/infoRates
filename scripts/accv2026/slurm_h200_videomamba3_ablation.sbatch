#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=vmamba3-ablate
#SBATCH --output=evaluations/accv2026/logs/vmamba3-ablate-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba3-ablate-%j.err
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
source .venv_mamba/bin/activate
export PYTHONPATH=src
export WANDB_MODE=online
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified

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
WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

for variant in ${VARIANTS}; do
  save_path="fine_tuned_models/accv2026_videomamba3_${MODEL_SIZE}_${variant}_${DATASET}_${INPUT_SIZE}r_f${NUM_FRAMES}_e${EPOCHS}_h200"
  echo "=== VideoMamba3 ${variant} ${DATASET} job=${SLURM_JOB_ID} ==="
  python scripts/accv2026/train_videomamba3.py \
    --dataset "${DATASET}" \
    --model-size "${MODEL_SIZE}" \
    --depth "${DEPTH}" \
    --mamba3-variant "${variant}" \
    --num-frames "${NUM_FRAMES}" \
    --input-size "${INPUT_SIZE}" \
    --ssm-d-state "${SSM_D_STATE}" \
    --ssm-expand "${SSM_EXPAND}" \
    --ssm-headdim "${SSM_HEADDIM}" \
    --ssm-mimo-rank "${SSM_MIMO_RANK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr 2e-5 \
    --weight-decay 0.05 \
    --num-workers 4 \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --no-pretrained \
    --save-path "${save_path}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "videomamba3-${MODEL_SIZE}-${variant}-${DATASET}-${INPUT_SIZE}r-f${NUM_FRAMES}-e${EPOCHS}-j${SLURM_JOB_ID}" \
    --wandb-tags accv2026 h200 "${DATASET}" videomamba3 "${variant}" "${MODEL_SIZE}" ablation official-mamba3
done

python scripts/accv2026/18_compile_videomamba3_results.py
