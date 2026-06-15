#!/usr/bin/env bash
# Retrain TimeSformer / ViViT / VideoMAE at 96/112/160px with proper pos-embed interpolation.
# Uses VERSION_SUFFIX=_v2 so results are stored in *_e10_v2_h200 dirs.
# 224px is excluded — native resolution results are already valid.
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
USER=wesleyferreiramaia
MAX_GPU=4           # gpu partition limit
MAX_CENV=4          # cenvalarc.gpu partition limit

gpu_count()  { squeue -u "$USER" -h -r -p gpu          2>/dev/null | wc -l; }
cenv_count() { squeue -u "$USER" -h -r -p cenvalarc.gpu 2>/dev/null | wc -l; }
total_count(){ squeue -u "$USER" -h -r                  2>/dev/null | wc -l; }

MODELS="timesformer vivit videomae"
DATASETS="autsl diving48 driveact ucf101 hmdb51 ssv2 epic_kitchens"
RESOLUTIONS="96 112 160"   # 224px excluded — already correct

submitted=0; skipped=0; failed=0

submit_one() {
    local MDL=$1 DS=$2 RES=$3
    local CKPT="${SCRATCH}/accv2026_${MDL}_${DS}_${RES}px_e10_v2_h200"
    if [[ -d "$CKPT" ]] && ls "$CKPT"/accv_meta.json 2>/dev/null | grep -q .; then
        echo "[SKIP] ${MDL}/${DS}@${RES}px — v2 checkpoint complete"
        return 1  # 1 = skipped
    fi
    if [[ "$RES" -le 112 ]]; then
        while [[ $(cenv_count) -ge $MAX_CENV ]]; do sleep 30; done
        PARTITION="cenvalarc.gpu"
    else
        while [[ $(gpu_count) -ge $MAX_GPU ]]; do sleep 30; done
        PARTITION="gpu"
    fi
    local JID
    JID=$(sbatch --partition="$PARTITION" \
        --export=ALL,MODEL="${MDL}",DATASET="${DS}",INPUT_SIZE="${RES}",VERSION_SUFFIX=_v2 \
        --output="${LOG}/${MDL}-v2-${DS}-${RES}px-%j.out" \
        --error="${LOG}/${MDL}-v2-${DS}-${RES}px-%j.err" \
        "$SBATCH" 2>&1) || {
        echo "[WARN] sbatch failed for ${MDL}/${DS}@${RES}px: $JID"
        return 2  # 2 = failed
    }
    echo "[$(date +%T)] $JID — ${MDL}/${DS}@${RES}px v2"
    sleep 3
    return 0
}

# Run two PARALLEL loops so cenvalarc saturation never blocks gpu:
#   Loop A (background) : 96px + 112px → cenvalarc.gpu
#   Loop B (foreground) : 160px         → gpu
# Both loops share the same submit_one/gpu_count/cenv_count functions.

sub_a=0; skp_a=0; fail_a=0
sub_b=0; skp_b=0; fail_b=0

# Loop A — small resolutions on cenvalarc
(
    for RES in 96 112; do
        for MDL in $MODELS; do
            for DS in $DATASETS; do
                submit_one "$MDL" "$DS" "$RES"
                rc=$?
                if   [[ $rc -eq 0 ]]; then ((sub_a++))  || true
                elif [[ $rc -eq 1 ]]; then ((skp_a++))  || true
                else                       ((fail_a++)) || true
                fi
            done
        done
    done
    echo "[cenv] Done. Submitted: $sub_a | Skipped: $skp_a | Failed: $fail_a"
) &
CENV_PID=$!

# Loop B — 160px on gpu
for MDL in $MODELS; do
    for DS in $DATASETS; do
        submit_one "$MDL" "$DS" 160
        rc=$?
        if   [[ $rc -eq 0 ]]; then ((sub_b++))  || true
        elif [[ $rc -eq 1 ]]; then ((skp_b++))  || true
        else                       ((fail_b++)) || true
        fi
    done
done
echo "[gpu]  Done. Submitted: $sub_b | Skipped: $skp_b | Failed: $fail_b"

wait $CENV_PID
echo "All loops complete."
