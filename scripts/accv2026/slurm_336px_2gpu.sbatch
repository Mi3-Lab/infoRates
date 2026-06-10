#!/usr/bin/env bash
#SBATCH --job-name=retrain-336px
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
# Data CSVs use paths relative to /scratch — must cd there for file resolution
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE=336
EPOCHS=10

# Data lives in /scratch, not /data
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
DATA_ROOT="${DATA_ROOTS[$DATASET]:-${SCRATCH_DATA}}"

source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

CHECKPOINT="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_${MODEL}_${DATASET}_336px_e${EPOCHS}_h200"

echo "=== 336px 2-GPU Retrain: ${MODEL} / ${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -10 || true

if [[ -f "${CHECKPOINT}/config.json" ]]; then
    echo "[SKIP] Checkpoint exists: ${CHECKPOINT}"
    exit 0
fi

if [[ "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" || "$MODEL" == "r2plus1d_18" ]]; then
    # 2× L40s (48GB each) — batch=64/GPU = 128 total, DDP
    export TORCH_COMPILE_DISABLE=1
    torchrun --nproc_per_node=2 /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_torchvision.py \
        --dataset   "${DATASET}" \
        --data-root "${DATA_ROOT}" \
        --model     "${MODEL}" \
        --epochs    "${EPOCHS}" \
        --batch-size 16 \
        --num-workers 8 \
        --input-size "${INPUT_SIZE}" \
        --save-path  "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-${MODEL}-${DATASET}-336px-2gpu-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 cenvalarc "${DATASET}" "${MODEL}" 336px 2gpu spatial-aliasing \
        --lr 1e-4 --weight-decay 0.05 --num-frames 16 \
        --ddp

elif [[ "$MODEL" == "slowfast_r50" ]]; then
    torchrun --nproc_per_node=2 /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_slowfast.py \
        --dataset "${DATASET}" \
        --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" \
        --batch-size 32 \
        --num-workers 8 \
        --input-size "${INPUT_SIZE}" \
        --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-slowfast-${DATASET}-336px-2gpu-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 cenvalarc "${DATASET}" slowfast 336px 2gpu spatial-aliasing \
        --ddp
fi

echo "=== DONE — $(date) ==="
