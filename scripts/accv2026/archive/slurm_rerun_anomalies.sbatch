#!/usr/bin/env bash
#SBATCH --job-name=retrain-rerun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# Re-run script for anomalous/truncated training jobs
# Runs on A100 (gpu partition, 40GB)
set -euo pipefail
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE="${INPUT_SIZE:?INPUT_SIZE must be set}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

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

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
CHECKPOINT="${CKPT_BASE}/accv2026_${MODEL}_${DATASET}_${INPUT_SIZE}px_e${EPOCHS}_h200"

echo "=== Retrain (rerun): ${MODEL} / ${DATASET} @ ${INPUT_SIZE}px (${EPOCHS} epochs) — job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "GPU: ${GPU_NAME}"

if [[ "$MODEL" == "r2plus1d_18" || "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    # A100 40GB: safe batch for CNNs at 96px
    [[ "$INPUT_SIZE" -le 96 ]] && BATCH=64 || ([[ "$INPUT_SIZE" -le 160 ]] && BATCH=48 || BATCH=32)
    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_torchvision.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" --model "${MODEL}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers "${NUM_WORKERS}" \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "rerun-${MODEL}-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" "${MODEL}" "${INPUT_SIZE}px" rerun \
        --lr 1e-4 --weight-decay 0.05 --num-frames 16 --ddp

elif [[ "$MODEL" == "videomamba" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    [[ "$GPU_NAME" == *"H200"* ]] && BATCH=48 || BATCH=16
    LR_SCHED="${LR_SCHED:-none}"
    WARMUP="${WARMUP:-0}"
    PATIENCE="${PATIENCE:-0}"
    EXTRA_ARGS=""
    [[ "$LR_SCHED" != "none" ]] && EXTRA_ARGS="$EXTRA_ARGS --lr-scheduler ${LR_SCHED}"
    [[ "$WARMUP" -gt 0 ]] && EXTRA_ARGS="$EXTRA_ARGS --warmup-epochs ${WARMUP}"
    [[ "$PATIENCE" -gt 0 ]] && EXTRA_ARGS="$EXTRA_ARGS --early-stopping-patience ${PATIENCE}"
    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_videomamba.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers "${NUM_WORKERS}" \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "rerun-videomamba-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" videomamba "${INPUT_SIZE}px" rerun \
        --lr 2e-5 --weight-decay 0.05 --ddp $EXTRA_ARGS

elif [[ "$MODEL" == "slowfast_r50" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    # A100 40GB: batch=32/GPU safe for SlowFast@112px (vs L40S original batch=16)
    [[ "$GPU_NAME" == *"H200"* ]] && BATCH=64 || BATCH=32
    torchrun --nproc_per_node=2 \
        /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_slowfast.py \
        --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers "${NUM_WORKERS}" \
        --input-size "${INPUT_SIZE}" --save-path "${CHECKPOINT}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "rerun-slowfast-${DATASET}-${INPUT_SIZE}px-j${SLURM_JOB_ID}" \
        --wandb-tags accv2026 "${DATASET}" slowfast "${INPUT_SIZE}px" rerun \
        --ddp
fi

echo "=== DONE — $(date) ==="
