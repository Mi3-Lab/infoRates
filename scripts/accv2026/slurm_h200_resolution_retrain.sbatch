#!/usr/bin/env bash
#SBATCH --job-name=retrain-res
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
# CWD must be /scratch so relative paths in manifests (data/AUTSL_data/...) resolve correctly
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
INPUT_SIZE="${INPUT_SIZE:?INPUT_SIZE must be set}"   # e.g. 112 or 224

# Batch sizes safe for L40s (48GB) and H200 (141GB) — whichever is assigned
# L40s = 48GB, H200 = 141GB. These values work on L40s; H200 can go higher.
if [[ "$MODEL" == "r3d_18" || "$MODEL" == "mc3_18" || "$MODEL" == "r2plus1d_18" ]]; then
    EPOCHS=10
    SCRIPT="train_torchvision"
    export TORCH_COMPILE_DISABLE=1  # Disable torch.compile for P3
    # Batch size: depends on GPU type (A100=40GB, L40=48GB, H200=141GB)
    if [[ "$INPUT_SIZE" -le 160 ]]; then
        BATCH_SIZE=64
    else  # 224px or 336px
        # Try to detect GPU type from environment or use conservative default
        if [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
            BATCH_SIZE=48    # A100: 40GB, batch=48 is tight @ 336px
        else
            BATCH_SIZE=64    # H200/L40: batch=64 is safe
        fi
    fi
elif [[ "$MODEL" == "slowfast_r50" ]]; then
    EPOCHS=10
    SCRIPT="train_slowfast"
    # SlowFast is memory-intensive due to dual-path
    if [[ "$INPUT_SIZE" -le 160 ]]; then
        BATCH_SIZE=48
    else  # 224px or 336px
        if [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
            BATCH_SIZE=24    # A100: very tight @ 336px
        else
            BATCH_SIZE=32    # H200/L40: safer
        fi
    fi
elif [[ "$MODEL" == "videomamba" ]]; then
    # Batch: 64 for small res (≤112px), 48 for 160/224px, 32 for 336px
    if [[ "$INPUT_SIZE" -le 112 ]]; then
        BATCH_SIZE=64
    elif [[ "$INPUT_SIZE" -le 224 ]]; then
        BATCH_SIZE=48
    else
        BATCH_SIZE=32
    fi
    EPOCHS=10
    SCRIPT="train_videomamba"
    source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate
    export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
else
    # timesformer, vivit, videomae
    BATCH_SIZE=48    # Transformers: safe on L40s (48GB)
    EPOCHS=10
    SCRIPT="train_transformers"
fi

if [[ "$MODEL" != "videomamba" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
fi

VERSION_SUFFIX="${VERSION_SUFFIX:-}"  # optional: "_v2", "_v3", etc.
CHECKPOINT="fine_tuned_models/accv2026_${MODEL}_${DATASET}_${INPUT_SIZE}px_e${EPOCHS}${VERSION_SUFFIX}_h200"

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
        python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_torchvision.py \
            "${COMMON_ARGS[@]}" \
            --lr 1e-4 --weight-decay 0.05 --num-frames 16
        ;;
    train_slowfast)
        # SlowFast has a separate training script
        python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_slowfast.py \
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
        python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_transformers.py \
            "${COMMON_ARGS[@]}" \
            --lr 2e-5 --weight-decay 0.05
        ;;
    train_videomamba)
        python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_videomamba.py \
            --dataset   "${DATASET}" \
            --epochs    "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers 16 \
            --input-size "${INPUT_SIZE}" \
            --save-path  "${CHECKPOINT}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "retrain-videomamba-${DATASET}-${INPUT_SIZE}px-e${EPOCHS}-j${SLURM_JOB_ID}" \
            --wandb-tags accv2026 h200 "${DATASET}" videomamba "${INPUT_SIZE}px" resolution-ablation spatial-aliasing \
            --lr 2e-5 --weight-decay 0.05 \
            --warmup-epochs 2 --lr-scheduler cosine
        ;;
esac

echo "=== DONE — $(date) ==="
