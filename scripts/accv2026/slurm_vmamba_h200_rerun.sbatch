#!/usr/bin/env bash
#SBATCH --job-name=vmamba-h200-fix
#SBATCH --partition=cenvalarc.gpu
#SBATCH --constraint=H200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# VideoMamba H200 rerun for anomalous checkpoints.
# Required env: DATASET, INPUT_SIZE, EPOCHS (default 10)
set -euo pipefail
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE="${INPUT_SIZE:?INPUT_SIZE must be set}"
EPOCHS="${EPOCHS:-10}"

SCRATCH_DATA="/scratch/wesleyferreiramaia/infoRates/data"
declare -A DATA_ROOTS=(
    [ucf101]="${SCRATCH_DATA}/UCF101_data"
    [ssv2]="${SCRATCH_DATA}/Something_data"
    [hmdb51]="${SCRATCH_DATA}/HMDB51_data"
    [diving48]="${SCRATCH_DATA}/Diving48_data"
    [autsl]="${SCRATCH_DATA}/AUTSL_data"
    [driveact]="${SCRATCH_DATA}/DriveAct_data"
    [epic_kitchens]="${SCRATCH_DATA}/EPIC_data"
)
DATA_ROOT="${DATA_ROOTS[$DATASET]}"
CHECKPOINT="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba_${DATASET}_${INPUT_SIZE}px_e10_h200"

echo "=== VideoMamba H200 rerun — ${DATASET} @ ${INPUT_SIZE}px (${EPOCHS}ep) — job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate

# H200: batch=48 for ≤224px, batch=32 for 336px
[[ "$INPUT_SIZE" -le 224 ]] && BATCH=48 || BATCH=32

torchrun --nproc_per_node=2 \
    /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_videomamba.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --save-path "${CHECKPOINT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --lr 2e-5 \
    --weight-decay 0.05 \
    --num-workers 16 \
    --input-size "${INPUT_SIZE}" \
    --ddp \
    --wandb-project inforates-accv2026 \
    --wandb-run-name "h200fix-videomamba-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
    --wandb-tags accv2026 "${DATASET}" videomamba "${INPUT_SIZE}px" h200-rerun

echo "=== DONE — $(date) ==="
