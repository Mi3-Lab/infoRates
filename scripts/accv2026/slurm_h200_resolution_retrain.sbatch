#!/usr/bin/env bash
#SBATCH --job-name=retrain-res
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE="${INPUT_SIZE:?INPUT_SIZE must be set}"   # e.g. 112 or 224

# Batch sizes tuned for H200 (141GB VRAM) per model family
if [[ "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" || "$MODEL" == "r2plus1d_18" ]]; then
    BATCH_SIZE=128
    EPOCHS=10
    SCRIPT="train_torchvision"
elif [[ "$MODEL" == "slowfast_r50" ]]; then
    BATCH_SIZE=32
    EPOCHS=10
    SCRIPT="train_slowfast"
elif [[ "$MODEL" == "videomamba" ]]; then
    BATCH_SIZE=32
    EPOCHS=10
    SCRIPT="train_videomamba"
    source .venv_mamba/bin/activate
    export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
else
    # timesformer, vivit, videomae
    BATCH_SIZE=32
    EPOCHS=10
    SCRIPT="train_transformers"
fi

if [[ "$MODEL" != "videomamba" ]]; then
    source .venv/bin/activate
fi

CHECKPOINT="fine_tuned_models/accv2026_${MODEL}_${DATASET}_${INPUT_SIZE}px_e${EPOCHS}_h200"

echo "=== Resolution Retrain: ${MODEL} / ${DATASET} @ ${INPUT_SIZE}px — job ${SLURM_JOB_ID} ==="
echo "    Script: ${SCRIPT}.py | Batch: ${BATCH_SIZE} | Epochs: ${EPOCHS}"
nvidia-smi | head -6 || true

if [[ -f "${CHECKPOINT}/config.json" ]]; then
    echo "[SKIP] Checkpoint exists: ${CHECKPOINT}"
    exit 0
fi

# Build common args
COMMON_ARGS=(
    --dataset   "${DATASET}"
    --model     "${MODEL}"
    --epochs    "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers 16
    --input-size "${INPUT_SIZE}"
    --save-path  "${CHECKPOINT}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "retrain-${MODEL}-${DATASET}-${INPUT_SIZE}px-e${EPOCHS}-j${SLURM_JOB_ID}"
    --wandb-tags accv2026 h200 "${DATASET}" "${MODEL}" "${INPUT_SIZE}px" resolution-ablation spatial-aliasing
)

case "$SCRIPT" in
    train_torchvision)
        python scripts/accv2026/train_torchvision.py \
            "${COMMON_ARGS[@]}" \
            --lr 1e-4 --weight-decay 0.05 --num-frames 16
        ;;
    train_slowfast)
        # SlowFast has a separate training script
        python scripts/accv2026/train_slowfast.py \
            --dataset "${DATASET}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers 16 \
            --save-path "${CHECKPOINT}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "retrain-slowfast-${DATASET}-${INPUT_SIZE}px-e${EPOCHS}-j${SLURM_JOB_ID}" \
            --wandb-tags accv2026 h200 "${DATASET}" slowfast "${INPUT_SIZE}px" resolution-ablation
        ;;
    train_transformers)
        python scripts/accv2026/train_transformers.py \
            "${COMMON_ARGS[@]}" \
            --lr 2e-5 --weight-decay 0.05
        ;;
    train_videomamba)
        python scripts/accv2026/train_videomamba.py \
            "${COMMON_ARGS[@]}" \
            --lr 2e-5 --weight-decay 0.05
        ;;
esac

echo "=== DONE — $(date) ==="
