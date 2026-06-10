#!/usr/bin/env bash
#SBATCH --job-name=retrain-h200
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# CRITICAL: cd /scratch so relative paths in CSVs resolve correctly
set -uo pipefail
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE="${INPUT_SIZE:?INPUT_SIZE must be set}"
EPOCHS=10

SCRATCH_DATA="/scratch/wesleyferreiramaia/infoRates/data"
declare -A DATA_ROOTS=(
    [ucf101]="${SCRATCH_DATA}/UCF101_data"
    [ssv2]="${SCRATCH_DATA}/Something_data"
    [hmdb51]="${SCRATCH_DATA}/HMDB51_data"
    [diving48]="${SCRATCH_DATA}/Diving48_data"
    [autsl]="${SCRATCH_DATA}/AUTSL_data"
    [driveact]="${SCRATCH_DATA}/DriveAct_data"
    [epic_kitchens]="${SCRATCH_DATA}/EPIC_data"
    [ego4d]="${SCRATCH_DATA}/Ego4D_data"
)
DATA_ROOT="${DATA_ROOTS[$DATASET]:-${SCRATCH_DATA}}"

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
CHECKPOINT="${CKPT_BASE}/accv2026_${MODEL}_${DATASET}_${INPUT_SIZE}px_e${EPOCHS}_h200"

echo "=== Retrain: ${MODEL} / ${DATASET} @ ${INPUT_SIZE}px — job ${SLURM_JOB_ID} ==="
nvidia-smi | grep -E "Name|H200|L40|A100" | head -4 || true

if [[ -f "${CHECKPOINT}/config.json" ]]; then
    echo "[SKIP] Already done: ${CHECKPOINT}"
    exit 0
fi

# Detect GPU type → set batch size accordingly (DDP: 2 GPUs)
# H200=141GB, L40s=48GB, A100=40GB per GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "GPU detected: ${GPU_NAME}"
IS_H200=false; IS_L40=false
[[ "$GPU_NAME" == *"H200"* ]] && IS_H200=true
[[ "$GPU_NAME" == *"L40"* ]] && IS_L40=true
if [[ "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" || "$MODEL" == "r2plus1d_18" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    export TORCH_COMPILE_DISABLE=1

    # H200: 64/GPU, L40s: 16/GPU (OOM at 32 @ 336px)
    if $IS_H200; then
        [[ "$INPUT_SIZE" -le 160 ]] && BATCH=96 || ([[ "$INPUT_SIZE" -le 224 ]] && BATCH=64 || BATCH=48)
    else
        BATCH=16  # L40s safe for all resolutions
    fi

    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_torchvision.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-${MODEL}-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" "${MODEL}" "${INPUT_SIZE}px" spatial-aliasing \
        --lr 1e-4 --weight-decay 0.05 --num-frames 16 --ddp

elif [[ "$MODEL" == "slowfast_r50" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

    if $IS_H200; then
        [[ "$INPUT_SIZE" -le 160 ]] && BATCH=64 || ([[ "$INPUT_SIZE" -le 224 ]] && BATCH=48 || BATCH=32)
    else
        BATCH=16  # L40s
    fi

    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_slowfast.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-slowfast-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" slowfast "${INPUT_SIZE}px" spatial-aliasing \
        --ddp

elif [[ "$MODEL" == "videomamba" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate

    if $IS_H200; then
        [[ "$INPUT_SIZE" -le 224 ]] && BATCH=48 || BATCH=32
    else
        BATCH=16
    fi

    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_videomamba.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-videomamba-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" videomamba "${INPUT_SIZE}px" spatial-aliasing \
        --lr 2e-5 --weight-decay 0.05 --ddp

else
    # timesformer, vivit, videomae
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

    if $IS_H200; then
        [[ "$INPUT_SIZE" -le 224 ]] && BATCH=32 || BATCH=16
    else
        BATCH=8   # L40s: Transformers são pesados
    fi

    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_transformers.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers 16 \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "retrain-${MODEL}-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" "${MODEL}" "${INPUT_SIZE}px" spatial-aliasing \
        --lr 2e-5 --weight-decay 0.05 --ddp
fi

echo "=== DONE — $(date) ==="
