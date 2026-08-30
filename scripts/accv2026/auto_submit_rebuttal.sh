#!/usr/bin/env bash
# Auto-submitter for the ACCV'26 rebuttal sweeps.
#
# Keeps the queue fed through all remaining stages without supervision, in the
# same style as master_auto_submit.sh. The Python runner skips (model, dataset)
# pairs that already have a summary.csv, so this is safe to restart at any point
# and will only fill gaps.
#
# Stages, in priority order:
#   1. matched  on kinetics400   — adds the dataset reviewer vdwp asked for, and
#                                  validates the SlowFast / VideoMamba K400 label
#                                  order (the two we could not check directly:
#                                  a wrong order shows up as ~0.25% top-1)
#   2. span     on all datasets  — the true sampling-rate axis (k fixed, window
#                                  width varies), the answer to "this is not
#                                  aliasing"
#   3. covstride on all datasets — submitted grid without repeat-last padding;
#                                  the gap to the published numbers is how much
#                                  of the cliff was the frozen tail
#
# Usage:
#   nohup bash scripts/accv2026/auto_submit_rebuttal.sh \
#         > evaluations/accv2026/logs/auto_submit_rebuttal.log 2>&1 &
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

# master_auto_submit.sh notes a hard QOS limit around 7; stay well under it so
# manual jobs can still be submitted alongside the daemon.
MAX_JOBS="${MAX_JOBS:-4}"
POLL="${POLL:-60}"

ALL_DATASETS="ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens kinetics400"

# Submit to both GPU partitions and let Slurm take whichever frees first.
# Pinning the Transformers to cenvalarc.gpu (L40s/H200) stalled them for over an
# hour at (Priority) with zero idle nodes there, while an A100 in `gpu` sat idle.
# At inference with batch 32 every model in the pool fits in 40GB, so there is no
# reason to pin. VideoMamba is excluded: mamba-ssm 2.3.1 in .venv_mamba is a
# CUDA 13 build while torch there is cu128, so `import mamba_ssm` fails with
# libcudart.so.13 missing. Re-add it here once that environment is repaired.
PARTITIONS="${PARTITIONS:-gpu,cenvalarc.gpu}"
MODELS_ALL="${MODELS_ALL:-r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Total jobs, used against the QOS submit cap (4) which counts everything.
n_jobs() { squeue -u "$(whoami)" --noheader 2>/dev/null | wc -l; }

# Only this daemon's sweep jobs. The TRA ablation runs for hours under its own
# job names, and draining on the *total* count would block the sweep stages
# behind it forever, so stage sequencing must look at "rb-" jobs alone.
n_sweep_jobs() {
    squeue -u "$(whoami)" --noheader -o "%j" 2>/dev/null | grep -c "^rb-" || true
}

wait_for_slot() {
    while true; do
        local n; n=$(n_jobs)
        [[ "$n" -lt "$MAX_JOBS" ]] && return 0
        log "  queue full (${n}/${MAX_JOBS}) — waiting for $1"
        sleep "$POLL"
    done
}

# Number of (model, dataset) pairs already finished for a mode.
done_count() {
    find "evaluations/accv2026/rebuttal_sweeps/$1" -name summary.csv 2>/dev/null | wc -l
}

submit_stage() {
    local mode=$1 datasets=$2
    log "=== stage: mode=${mode} ==="
    for model in ${MODELS_ALL}; do
        # Retry until this model is actually queued. Moving on after a failed
        # sbatch would silently drop it from the stage: QOSMaxSubmitJobPerUser
        # can reject a submission even when squeue looks under the limit, and a
        # skipped model leaves a hole that nothing later fills.
        local out attempt=0
        while true; do
            wait_for_slot "${mode}/${model}"
            out=$(MODE="${mode}" MODELS="${model}" DATASETS="${datasets}" \
                  sbatch --partition="${PARTITIONS}" \
                         --time=1-00:00:00 \
                         --job-name="rb-${mode}-${model}" \
                         --export=ALL \
                         scripts/accv2026/slurm_rebuttal_sweep.sbatch 2>&1)
            if [[ "${out}" == *"Submitted batch job"* ]]; then
                log "  submitted ${mode}/${model} — ${out##* }"
                break
            fi
            attempt=$((attempt + 1))
            log "  [retry ${attempt}] ${mode}/${model}: ${out}"
            sleep "${POLL}"
        done
        sleep 2
    done
    log "=== stage ${mode}: all jobs submitted ==="
}

# Block until everything currently queued has drained, so stages do not
# interleave and each one's results are complete before the next is analysed.
drain() {
    while [[ "$(n_sweep_jobs)" -gt 0 ]]; do
        log "  draining: $(n_sweep_jobs) sweep jobs left (total $(n_jobs)) | "\
"matched=$(done_count matched) span=$(done_count span) covstride=$(done_count covstride)"
        sleep "$POLL"
    done
}

log "daemon start — MAX_JOBS=${MAX_JOBS}, poll=${POLL}s"
log "waiting for the in-flight matched jobs to finish first"
drain
log "matched (7 datasets) done: $(done_count matched) pairs"

submit_stage matched "kinetics400"
drain
log "matched total: $(done_count matched) pairs"

submit_stage span "${ALL_DATASETS}"
drain
log "span total: $(done_count span) pairs"

# Anti-aliasing intervention: same k, same span, point-sampled vs temporally
# box-filtered. This is the only experiment here that can test the Nyquist
# hypothesis causally -- if the loss is aliasing, removing above-Nyquist energy
# must recover accuracy. Ranked ahead of covstride because covstride mainly
# serves pxtb, who states the paper is beyond rebuttal repair.
submit_stage antialias "${ALL_DATASETS}"
drain
log "antialias total: $(done_count antialias) pairs"

submit_stage covstride "${ALL_DATASETS}"
drain
log "covstride total: $(done_count covstride) pairs"

log "ALL STAGES COMPLETE"
log "  matched=$(done_count matched)  span=$(done_count span)  covstride=$(done_count covstride)"
