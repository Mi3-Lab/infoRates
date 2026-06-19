#!/usr/bin/env bash
# P3 FineGym — Retrain the 6 checkpoints that OOMed in the main script's retry block.
#
# Missing checkpoints (OOM during retry pass, insufficient CUDA pool drain):
#   vivit       @ 224px  (native size, no pos_embed issue)
#   vivit       @ 336px  (non-native — model_factory.py bicubic fix applies)
#   videomamba  @ 224px  (native size)
#   videomamba  @ 336px  (non-native — VideoMamba has its own interpolation)
#   r2plus1d_18 @ 336px  (CNN, BatchNorm — needs bs≥32, expandable_segments)
#   slowfast_r50@ 336px  (dual-path CNN, bs=32 safe at 336px)
#
# Root cause of original OOM: retry block had only ~8s gaps between long runs,
# insufficient to drain CUDAGraph private pools (10-20GB).  This script uses
# 90s sleep between every run to ensure full pool release.
#
# Usage:
#   cd /mnt/datasets/infoRates
#   nohup bash scripts/accv2026/run_p3_retrain_finegym_oom.sh \
#         > evaluations/accv2026/logs/p3_retrain_finegym_oom.log 2>&1 &
#   tail -f evaluations/accv2026/logs/p3_retrain_finegym_oom.log

set -uo pipefail
cd /mnt/datasets/infoRates
export PYTHONPATH=src

if [[ -d "/scratch/wesleyferreiramaia" ]]; then
    export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
    export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
fi
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online

# Expandable segments reduces fragmentation for CNN/SlowFast 336px
# (bs=32 at 336px fills ~47GB activations, but fragmentation can push it over)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=".venv"
CKPT_BASE="fine_tuned_models"
DATASET="finegym"
EPOCHS=10

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3-OOM | $*"; }

is_done() {
    local model=$1 res=$2
    [[ -f "${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200/config.json" ]]
}

# 90s drain — main script OOMs were caused by only ~8s between runs.
# Fixpos script ran cleanly with 30s; using 90s here as extra margin since
# these follow very long runs (videomamba×5 can leave large pools).
cuda_drain() { sleep 90; rm -f /dev/shm/torch_* /dev/shm/sem.mp-* 2>/dev/null || true; }

# ─── Transformer (vivit) ─────────────────────────────────────────────────────
run_transformer() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    if is_done "$model" "$res"; then
        log "SKIP ${model}@${res}px (already done)"
        return 0
    fi

    # ViViT has 32 frames × (res/16)² patches: factorised attention is very
    # memory-heavy.  bs=32 at 224px needs ~103GiB (OOM); bs=8 ≈ 26GiB.
    # bs=24 at 336px needs ~109GiB (OOM); bs=4 ≈ 18GiB.
    local bs=32
    [[ $res -le 112 ]] && bs=64
    [[ $res -eq 224 ]] && bs=8
    [[ $res -ge 336 ]] && bs=4

    log "START ${model} @ ${res}px  batch=${bs}"
    source "${VENV}/bin/activate"
    cuda_drain

    # vivit@336: model_factory.py bicubic pos_embed fix is already active —
    # train_transformers.py calls ModelFactory.load_model() which interpolates
    # from native 224px when target_size != native_size and no checkpoint exists.
    if ! python scripts/accv2026/train_transformers.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --lr            2e-5 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${res}px-e${EPOCHS}-oom-retry" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing oom-retry; then
        log "ERROR ${model} @ ${res}px — training failed"
        return 1
    fi
    log "DONE  ${model} @ ${res}px"
}

# ─── VideoMamba ───────────────────────────────────────────────────────────────
run_videomamba() {
    local res=$1
    local model="videomamba"
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    if is_done "$model" "$res"; then
        log "SKIP ${model}@${res}px (already done)"
        return 0
    fi

    local bs=48
    [[ $res -le 112 ]] && bs=64
    [[ $res -ge 336 ]] && bs=24

    log "START ${model} @ ${res}px  batch=${bs}"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_videomamba.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --lr            2e-5 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-videomamba-${DATASET}-${res}px-e${EPOCHS}-oom-retry" \
        --wandb-tags    accv2026 "${DATASET}" videomamba "${res}px" resolution-ablation spatial-aliasing oom-retry; then
        log "ERROR ${model} @ ${res}px — training failed"
        return 1
    fi
    log "DONE  ${model} @ ${res}px"
}

# ─── CNN (r2plus1d_18) ────────────────────────────────────────────────────────
run_cnn() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    if is_done "$model" "$res"; then
        log "SKIP ${model}@${res}px (already done)"
        return 0
    fi

    # r2plus1d_18@336: split convolutions are 2× memory vs pure 3D; bs=32 safe at 336px
    local bs=32

    log "START ${model} @ ${res}px  batch=${bs}  [expandable_segments]"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_torchvision.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --num-frames    16 \
        --lr            1e-4 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${res}px-e${EPOCHS}-oom-retry" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing oom-retry; then
        log "ERROR ${model} @ ${res}px — training failed"
        return 1
    fi
    log "DONE  ${model} @ ${res}px"
}

# ─── SlowFast ─────────────────────────────────────────────────────────────────
run_slowfast() {
    local res=$1
    local model="slowfast_r50"
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"
    if is_done "$model" "$res"; then
        log "SKIP ${model}@${res}px (already done)"
        return 0
    fi

    # 336px: bs=32 (main script's retry block used bs=32; bs=48 OOMed)
    local bs=32

    log "START ${model} @ ${res}px  batch=${bs}  [expandable_segments]"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_slowfast.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    "${bs}" \
        --num-workers   12 \
        --input-size    "${res}" \
        --save-path     "${ckpt}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-slowfast-${DATASET}-${res}px-e${EPOCHS}-oom-retry" \
        --wandb-tags    accv2026 "${DATASET}" slowfast "${res}px" resolution-ablation spatial-aliasing oom-retry; then
        log "ERROR ${model} @ ${res}px — training failed"
        return 1
    fi
    log "DONE  ${model} @ ${res}px"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
log "=== P3 FineGym OOM retry: 6 missing checkpoints ==="
log "CUDA pool drain: 90s between runs (main script had ~8s — root cause of original OOM)"
log ""

failed=()

# vivit@224 — native size, no pos_embed issue, should be fast (~20min)
run_transformer vivit 224 || failed+=(vivit@224)

# vivit@336 — non-native, bicubic pos_embed fix active in model_factory.py
run_transformer vivit 336 || failed+=(vivit@336)

# videomamba@224 — native size
run_videomamba 224 || failed+=(videomamba@224)

# videomamba@336 — non-native, VideoMamba has its own pos_embed interpolation
run_videomamba 336 || failed+=(videomamba@336)

# r2plus1d_18@336 — CNN, bs=32, expandable_segments
run_cnn r2plus1d_18 336 || failed+=(r2plus1d_18@336)

# slowfast_r50@336 — dual-path CNN, bs=32, expandable_segments
run_slowfast 336 || failed+=(slowfast_r50@336)

log ""
if [[ ${#failed[@]} -eq 0 ]]; then
    log "=== ALL 6 runs complete ==="
else
    log "=== Done with ${#failed[@]} failure(s): ${failed[*]} ==="
fi
log "Next: python scripts/accv2026/eval_p3_retrained.py --dataset finegym"
