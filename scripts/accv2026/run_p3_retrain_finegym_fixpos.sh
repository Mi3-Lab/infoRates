#!/usr/bin/env bash
# P3 FineGym — RETRAIN transformers at non-native resolutions with FIXED pos_embed.
#
# Background: model_factory.py previously passed ignore_mismatched_sizes=True which discards
# pretrained pos_embed at non-native resolutions and randomly reinitializes it.  Result:
# timesformer/vivit/videomae trained at 96/112/160px start with zero spatial priors.
#
# Fix applied 2026-06-11: bicubic interpolation from native 224px → target resolution,
# identical to the approach VideoMamba already used.  This script overwrites the 10 affected
# checkpoints (9 already done + timesformer@336 which will have finished with the bug):
#
#   timesformer @ 96, 112, 160, 336px
#   vivit       @ 96, 112, 160px       (vivit@336 was trained by main script with the fix)
#   videomae    @ 96, 112, 160px       (videomae@336 was trained by main script with the fix)
#
# timesformer@224 / vivit@224 / videomae@224 are unaffected (native size, no interpolation).
#
# Usage:
#   cd /mnt/datasets/infoRates
#   nohup bash scripts/accv2026/run_p3_retrain_finegym_fixpos.sh \
#         > evaluations/accv2026/logs/p3_retrain_finegym_fixpos.log 2>&1 &
#   tail -f evaluations/accv2026/logs/p3_retrain_finegym_fixpos.log

set -uo pipefail
cd /mnt/datasets/infoRates
export PYTHONPATH=src

if [[ -d "/scratch/wesleyferreiramaia" ]]; then
    export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
    export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
fi
export WANDB_PROJECT=inforates-accv2026
export WANDB_MODE=online

VENV=".venv"
CKPT_BASE="fine_tuned_models"
DATASET="finegym"
EPOCHS=10

# PID of the main P3 retraining script launched on 2026-06-10
MAIN_SCRIPT_PID=2589955

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3-FixPos | $*"; }

cuda_drain() { sleep 30; }

# ─── Wait for main script ────────────────────────────────────────────────────
if kill -0 "$MAIN_SCRIPT_PID" 2>/dev/null; then
    log "Waiting for main script (PID $MAIN_SCRIPT_PID) to finish first..."
    while kill -0 "$MAIN_SCRIPT_PID" 2>/dev/null; do sleep 60; done
    log "Main script done — starting pos_embed fix retraining"
    # Extra drain: main script may have left CUDAGraph pools
    log "Draining CUDA memory pools (sleep 90s)..."
    sleep 90
else
    log "Main script (PID $MAIN_SCRIPT_PID) already finished"
fi

# Clean up orphaned shared memory from any lingering workers
rm -f /dev/shm/torch_* /dev/shm/sem.mp-* 2>/dev/null || true

# ─── Retrain function ────────────────────────────────────────────────────────
run_transformer_fixpos() {
    local model=$1 res=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e${EPOCHS}_h200"

    # Remove buggy checkpoint so training starts fresh with fixed pos_embed
    if [[ -d "${ckpt}" ]]; then
        log "Removing buggy checkpoint: ${ckpt}"
        rm -rf "${ckpt}"
    fi

    # Clear stale eval cache for this checkpoint
    local eval_dir="evaluations/accv2026/p3_retrained/${model}_${DATASET}"
    local eval_cache="${eval_dir}/res${res}_retrained_summary.csv"
    local eval_samples="${eval_dir}/res${res}_retrained_samples.csv"
    if [[ -f "$eval_cache" ]]; then
        rm -f "$eval_cache" "$eval_samples"
        log "Cleared eval cache: ${eval_cache}"
    fi

    # Batch sizes — same as main script (LayerNorm: not sensitive to small bs)
    local bs=32
    [[ $res -le 112 ]] && bs=64
    [[ $res -ge 336 ]] && bs=24

    log "START ${model} @ ${res}px  batch=${bs}  [bicubic pos_embed fix]"
    source "${VENV}/bin/activate"
    cuda_drain

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
        --wandb-run-name "p3-${model}-${DATASET}-${res}px-e${EPOCHS}-fixpos" \
        --wandb-tags    accv2026 "${DATASET}" "${model}" "${res}px" resolution-ablation spatial-aliasing fixpos; then
        log "ERROR ${model} @ ${res}px — training failed, continuing to next run"
        return 1
    fi

    log "DONE  ${model} @ ${res}px"
}

# ─── Main ────────────────────────────────────────────────────────────────────
log "=== P3 FineGym pos_embed fix: 10 runs (4+3+3) ==="
log "Fix: bicubic interpolation from 224px replaces random reinit at non-native sizes"
log ""

failed=()

# timesformer: 4 non-native resolutions
for res in 96 112 160 336; do
    run_transformer_fixpos timesformer "$res" || failed+=("timesformer@${res}px")
done

# vivit: 3 non-native resolutions (vivit@336 trained by main script with fix already applied)
for res in 96 112 160; do
    run_transformer_fixpos vivit "$res" || failed+=("vivit@${res}px")
done

# videomae: 3 non-native resolutions (videomae@336 trained by main script with fix already applied)
for res in 96 112 160; do
    run_transformer_fixpos videomae "$res" || failed+=("videomae@${res}px")
done

log ""
if [[ ${#failed[@]} -eq 0 ]]; then
    log "=== ALL 10 runs complete ==="
else
    log "=== Done with ${#failed[@]} failure(s): ${failed[*]} ==="
fi
log "Next: python scripts/accv2026/eval_p3_retrained.py --dataset finegym"
