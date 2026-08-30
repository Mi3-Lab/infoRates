#!/usr/bin/env bash
# Evaluate the TRA ablation checkpoints once their training finishes.
#
# Training accuracy tells us nothing about the question the ablation asks: the
# validation loader is unaugmented, so val_acc is measured under dense sampling.
# What matters is each checkpoint's behaviour across the stride grid, which is
# why every arm is re-run through the covstride sweep and compared against the
# published baseline checkpoint.
#
# The script polls for finished checkpoints rather than assuming an order, so it
# can be started at any point and will pick arms up as they complete.
#
# Usage:
#   nohup bash scripts/accv2026/auto_eval_tra.sh \
#         > evaluations/accv2026/logs/auto_eval_tra.log 2>&1 &
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_JOBS="${MAX_JOBS:-4}"      # QOS submit cap
POLL="${POLL:-300}"
MODE="${MODE:-covstride_pad}"
DATASET="${DATASET:-autsl}"
ARMS="${ARMS:-paper fixed}"
MODELS="${MODELS:-timesformer r3d_18}"
CKPT_ROOT="${CKPT_ROOT:-/data/wesleyferreiramaia/infoRates/fine_tuned_models}"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT_HOURS:-24} * 3600 ))

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
n_jobs() { squeue -u "$(whoami)" --noheader 2>/dev/null | wc -l; }

# Training job names abbreviate the model, so map back explicitly. Matching on
# the arm alone would block timesformer/paper while r3d/paper is still training,
# since "^tra-.*-paper$" matches both.
short_name() {
    case "$1" in
        timesformer)  echo tsf ;;
        r3d_18)       echo r3d ;;
        mc3_18)       echo mc3 ;;
        r2plus1d_18)  echo r2p1d ;;
        slowfast_r50) echo sf ;;
        videomae)     echo vmae ;;
        vivit)        echo vivit ;;
        videomamba)   echo vmamba ;;
        *)            echo "$1" ;;
    esac
}

# A checkpoint is ready when its directory exists AND that specific (model, arm)
# training job is no longer queued or running -- otherwise we would evaluate a
# model saved at an intermediate best epoch, since save_pretrained writes on
# every improvement.
training_active() {
    local job="tra-$(short_name "$1")-$2"
    squeue -u "$(whoami)" --noheader -o "%j" 2>/dev/null | grep -qx "${job}"
}

submitted_marker() { echo "evaluations/accv2026/logs/.tra_eval_${1}_${2}.done"; }

for_each_arm() {
    for model in ${MODELS}; do
        for arm in ${ARMS}; do
            local ckpt="${CKPT_ROOT}/accv2026_${model}_${DATASET}_tra_${arm}"
            local marker; marker=$(submitted_marker "${model}" "${arm}")
            [[ -f "${marker}" ]] && continue
            [[ -d "${ckpt}" ]] || continue
            training_active "${model}" "${arm}" && continue

            while [[ "$(n_jobs)" -ge "${MAX_JOBS}" ]]; do
                log "  queue full — waiting to submit ${model}/${arm}"
                sleep "${POLL}"
            done

            local out
            out=$(MODE="${MODE}" MODELS="${model}" DATASETS="${DATASET}" \
                  EXTRA_ARGS="--checkpoint ${ckpt} --tag tra_${arm}" \
                  sbatch --partition=gpu,cenvalarc.gpu \
                         --time=08:00:00 \
                         --job-name="ev-tra-${arm}-${model}" \
                         --export=ALL \
                         scripts/accv2026/slurm_rebuttal_sweep.sbatch 2>&1)
            if [[ "${out}" == *"Submitted batch job"* ]]; then
                log "  submitted eval ${model}/${arm} — ${out##* }"
                touch "${marker}"
            else
                log "  [retry later] ${model}/${arm}: ${out}"
            fi
        done
    done
}

log "TRA eval watcher start — mode=${MODE} dataset=${DATASET}"
while [[ "$(date +%s)" -lt "${DEADLINE}" ]]; do
    for_each_arm
    pending=0
    for model in ${MODELS}; do
        for arm in ${ARMS}; do
            [[ -f "$(submitted_marker "${model}" "${arm}")" ]] || pending=1
        done
    done
    [[ "${pending}" -eq 0 ]] && { log "all arms submitted"; break; }
    sleep "${POLL}"
done
log "watcher done"
