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
    if [[ "$INPUT_SIZE" -le 48 ]]; then
        BATCH_SIZE=128   # 48px: avoid DataLoader overhead with excessive batch
    elif [[ "$INPUT_SIZE" -le 160 ]]; then
        BATCH_SIZE=64
    else  # 224px
        if [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
            BATCH_SIZE=48
        else
            BATCH_SIZE=64
        fi
    fi
elif [[ "$MODEL" == "slowfast_r50" ]]; then
    EPOCHS=10
    SCRIPT="train_slowfast"
    if [[ "$INPUT_SIZE" -le 48 ]]; then
        BATCH_SIZE=128   # 48px dual-path still fits large batch
    elif [[ "$INPUT_SIZE" -le 160 ]]; then
        BATCH_SIZE=48
    else  # 224px
        if [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
            BATCH_SIZE=24
        else
            BATCH_SIZE=32
        fi
    fi
elif [[ "$MODEL" == "videomamba" ]]; then
    # Detect GPU memory at runtime to set safe batch size
    GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    if [[ "$INPUT_SIZE" -le 48 ]]; then
        BATCH_SIZE=128   # 48px: 3×3=9 tokens, very low memory
    elif [[ "$INPUT_SIZE" -le 112 ]]; then
        BATCH_SIZE=64
    elif [[ "$INPUT_SIZE" -le 224 ]]; then
        # L40S=44GB OOMs at batch=48 — use 16; H200=141GB handles 48
        if [[ "$GPU_MEM_MIB" -gt 80000 ]]; then
            BATCH_SIZE=48
        else
            BATCH_SIZE=16
        fi
    else
        BATCH_SIZE=32
    fi
    EPOCHS=10
    SCRIPT="train_videomamba"
    source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate
    export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
else
    # timesformer, vivit, videomae
    if [[ "$INPUT_SIZE" -le 48 ]]; then
        BATCH_SIZE=96    # 48px: patch_size=16 → 3×3=9 spatial tokens, low memory
    else
        BATCH_SIZE=48    # Transformers: safe on L40s (48GB)
    fi
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

# videomamba saves accv_meta.json (not config.json); check both
SKIP_MARKER="${CHECKPOINT}/config.json"
if [[ "$MODEL" == "videomamba" ]]; then
    SKIP_MARKER="${CHECKPOINT}/accv_meta.json"
fi
if [[ -f "${SKIP_MARKER}" ]]; then
    echo "[SKIP] Checkpoint exists: ${CHECKPOINT}"
    exit 0
fi

# SSv2 webm files cause FFMPEG EAGAIN errors with high worker counts;
# reduce to 4 to avoid RuntimeError after 8 decode attempts.
NUM_WORKERS=16
[[ "$DATASET" == "ssv2" ]] && NUM_WORKERS=4

# Build common args
COMMON_ARGS=(
    --dataset   "${DATASET}"
    --model     "${MODEL}"
    --epochs    "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
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
            --num-workers "${NUM_WORKERS}" \
            --input-size "${INPUT_SIZE}" \
            --save-path "${CHECKPOINT}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "retrain-slowfast-${DATASET}-${INPUT_SIZE}px-e${EPOCHS}-j${SLURM_JOB_ID}" \
            --wandb-tags accv2026 h200 "${DATASET}" slowfast "${INPUT_SIZE}px" resolution-ablation
        ;;
    train_transformers)
        # For sub-native resolutions, start from the already-adapted 224px checkpoint
        # (same model+dataset). This avoids learning AUTSL from scratch at 96px which
        # collapses to ~22% due to insufficient spatial information at init.
        NATIVE_CKPT="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_${MODEL}_${DATASET}_224px_e10_h200"
        if [[ "${INPUT_SIZE}" != "224" && -d "${NATIVE_CKPT}" && -f "${NATIVE_CKPT}/accv_meta.json" ]]; then
            PRETRAINED_ARG="--pretrained-from ${NATIVE_CKPT}"
            echo "[Init] Starting from 224px checkpoint: ${NATIVE_CKPT}"
        else
            PRETRAINED_ARG=""
            echo "[Init] Starting from HuggingFace pretrained (${MODEL_ID:-HF hub})"
        fi
        python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_transformers.py \
            "${COMMON_ARGS[@]}" \
            --lr 2e-5 --weight-decay 0.05 \
            ${PRETRAINED_ARG}
        ;;
    train_videomamba)
        # For all datasets: init from the v1 224px checkpoint so BiMamba already
        # knows the target domain. Without this, BiMamba is randomly initialized
        # (K400 ckpt does not train the SSM layers) and 10 epochs at lr=2e-5 causes
        # severe overfitting (train_loss drops, val_loss rises — no generalization).
        # VMAMBA_PRETRAINED_FROM_OVERRIDE: explicit path override (e.g. use v2 160px as init).
        VMAMBA_PRETRAINED_ARG=""
        if [[ -n "${VMAMBA_PRETRAINED_FROM_OVERRIDE:-}" && -d "${VMAMBA_PRETRAINED_FROM_OVERRIDE}" && -f "${VMAMBA_PRETRAINED_FROM_OVERRIDE}/accv_meta.json" ]]; then
            VMAMBA_PRETRAINED_ARG="--pretrained-from ${VMAMBA_PRETRAINED_FROM_OVERRIDE}"
            echo "[Init] VideoMamba ${DATASET}@${INPUT_SIZE}px: OVERRIDE pretrained-from ${VMAMBA_PRETRAINED_FROM_OVERRIDE}"
        else
            V1_224="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba_${DATASET}_224px_e10_h200"
            if [[ -d "$V1_224" && -f "$V1_224/accv_meta.json" ]]; then
                VMAMBA_PRETRAINED_ARG="--pretrained-from ${V1_224}"
                echo "[Init] VideoMamba ${DATASET}@${INPUT_SIZE}px: initializing from v1 224px checkpoint"
            else
                echo "[WARN] No v1 224px checkpoint found for ${DATASET} — falling back to K400 init"
            fi
        fi
        python /data/wesleyferreiramaia/infoRates/scripts/accv2026/train_videomamba.py \
            --dataset   "${DATASET}" \
            --epochs    "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${NUM_WORKERS}" \
            --input-size "${INPUT_SIZE}" \
            --save-path  "${CHECKPOINT}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "retrain-videomamba-${DATASET}-${INPUT_SIZE}px-e${EPOCHS}-j${SLURM_JOB_ID}" \
            --wandb-tags accv2026 h200 "${DATASET}" videomamba "${INPUT_SIZE}px" resolution-ablation spatial-aliasing \
            --lr 2e-5 --weight-decay 0.05 \
            ${VMAMBA_PRETRAINED_ARG}
        ;;
esac

echo "=== DONE — $(date) ==="
