#!/usr/bin/env bash
# P3 FineGym — Resolution sweep at 48px (half of 96px).
#
# Rationale: the paper studies spatial aliasing analogous to Shannon-Nyquist.
# 48px is the most aggressive sub-Nyquist point, completing the curve:
#   48 → 96 → 112 → 160 → 224px
#
# At 48px, patch_size=16 → 3×3 = 9 spatial patches per frame for transformers.
# CNNs have no patch constraint and handle 48px via spatial pooling.
#
# Batch sizes: 48px clips are tiny — can use large batches.
#   CNNs:         bs=256  (well above BatchNorm minimum)
#   SlowFast:     bs=128
#   Transformers: bs=128  (9 patches/frame, trivial memory)
#   VideoMamba:   bs=128
#
# num_workers=4 at 48px: clips are tiny (~3MB each), 4 workers is plenty
# and avoids /dev/shm pressure from large prefetch buffers.
#
# Usage:
#   cd /mnt/datasets/infoRates
#   nohup bash scripts/accv2026/run_p3_retrain_finegym_48px.sh \
#         > evaluations/accv2026/logs/p3_retrain_finegym_48px.log 2>&1 &
#   tail -f evaluations/accv2026/logs/p3_retrain_finegym_48px.log

set -uo pipefail
cd /mnt/datasets/infoRates
export PYTHONPATH=src

if [[ -d "/scratch/wesleyferreiramaia" ]]; then
    export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
    export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
fi
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=".venv"
CKPT_BASE="fine_tuned_models"
DATASET="finegym"
EPOCHS=10
RES=48

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3-48px | $*"; }

is_done() {
    local model=$1
    [[ -f "${CKPT_BASE}/accv2026_${model}_${DATASET}_${RES}px_e${EPOCHS}_h200/config.json" ]]
}

cuda_drain() {
    sleep 60
    rm -f /dev/shm/torch_* /dev/shm/sem.mp-* 2>/dev/null || true
}

# ─── CNN ─────────────────────────────────────────────────────────────────────
run_cnn() {
    local model=$1
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${RES}px_e${EPOCHS}_h200"
    is_done "$model" && { log "SKIP ${model}@${RES}px (done)"; return 0; }

    log "START ${model} @ ${RES}px  batch=256"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_torchvision.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    256 \
        --num-workers   4 \
        --input-size    "${RES}" \
        --save-path     "${ckpt}" \
        --num-frames    16 \
        --lr            1e-4 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${RES}px-e${EPOCHS}" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${RES}px" resolution-ablation spatial-aliasing; then
        log "ERROR ${model} @ ${RES}px"
        return 1
    fi
    log "DONE  ${model} @ ${RES}px"
}

# ─── SlowFast ─────────────────────────────────────────────────────────────────
run_slowfast() {
    local ckpt="${CKPT_BASE}/accv2026_slowfast_r50_${DATASET}_${RES}px_e${EPOCHS}_h200"
    is_done "slowfast_r50" && { log "SKIP slowfast_r50@${RES}px (done)"; return 0; }

    log "START slowfast_r50 @ ${RES}px  batch=128"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_slowfast.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    128 \
        --num-workers   4 \
        --input-size    "${RES}" \
        --save-path     "${ckpt}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-slowfast-${DATASET}-${RES}px-e${EPOCHS}" \
        --wandb-tags    accv2026 "${DATASET}" slowfast "${RES}px" resolution-ablation spatial-aliasing; then
        log "ERROR slowfast_r50 @ ${RES}px"
        return 1
    fi
    log "DONE  slowfast_r50 @ ${RES}px"
}

# ─── Transformer ──────────────────────────────────────────────────────────────
run_transformer() {
    local model=$1
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${RES}px_e${EPOCHS}_h200"
    is_done "$model" && { log "SKIP ${model}@${RES}px (done)"; return 0; }

    # 48px → 3×3 = 9 patches/frame — trivially small, bs=128 fits easily
    log "START ${model} @ ${RES}px  batch=128  [bicubic pos_embed fix]"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_transformers.py \
        --dataset       "${DATASET}" \
        --model         "${model}" \
        --epochs        "${EPOCHS}" \
        --batch-size    128 \
        --num-workers   4 \
        --input-size    "${RES}" \
        --save-path     "${ckpt}" \
        --lr            2e-5 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-${model}-${DATASET}-${RES}px-e${EPOCHS}" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${RES}px" resolution-ablation spatial-aliasing; then
        log "ERROR ${model} @ ${RES}px"
        return 1
    fi
    log "DONE  ${model} @ ${RES}px"
}

# ─── VideoMamba ───────────────────────────────────────────────────────────────
run_videomamba() {
    local ckpt="${CKPT_BASE}/accv2026_videomamba_${DATASET}_${RES}px_e${EPOCHS}_h200"
    is_done "videomamba" && { log "SKIP videomamba@${RES}px (done)"; return 0; }

    log "START videomamba @ ${RES}px  batch=128"
    source "${VENV}/bin/activate"
    cuda_drain

    if ! python scripts/accv2026/train_videomamba.py \
        --dataset       "${DATASET}" \
        --epochs        "${EPOCHS}" \
        --batch-size    128 \
        --num-workers   4 \
        --input-size    "${RES}" \
        --save-path     "${ckpt}" \
        --lr            2e-5 \
        --weight-decay  0.05 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "p3-videomamba-${DATASET}-${RES}px-e${EPOCHS}" \
        --wandb-tags    accv2026 "${DATASET}" videomamba "${RES}px" resolution-ablation spatial-aliasing; then
        log "ERROR videomamba @ ${RES}px"
        return 1
    fi
    log "DONE  videomamba @ ${RES}px"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
log "=== P3 FineGym @ 48px — 8 models ==="
log "Curve: 48 → 96 → 112 → 160 → 224px"
log "48px: 3×3=9 spatial patches/frame for transformers; trivial for CNNs"
log ""

failed=()

# CNNs (fast at 48px — each epoch < 5min)
for model in r3d_18 mc3_18 r2plus1d_18; do
    run_cnn "$model" || failed+=("${model}@${RES}px")
done

# SlowFast
run_slowfast || failed+=("slowfast_r50@${RES}px")

# Transformers (bicubic pos_embed fix active in model_factory.py)
for model in timesformer vivit videomae; do
    run_transformer "$model" || failed+=("${model}@${RES}px")
done

# VideoMamba
run_videomamba || failed+=("videomamba@${RES}px")

log ""
if [[ ${#failed[@]} -eq 0 ]]; then
    log "=== ALL 8 runs complete ==="
else
    log "=== Done with ${#failed[@]} failure(s): ${failed[*]} ==="
fi
log "Next: python scripts/accv2026/eval_p3_retrained.py --dataset finegym"
