#!/usr/bin/env bash
# Submit the three rebuttal sweeps (matched / span / covstride).
#
# One job per (mode, model): each job loads the model once and walks every
# dataset, which is how the published sweep amortized checkpoint loading.
# Completed (model, dataset) pairs are skipped by the Python script, so
# re-running this after a partial failure only fills the gaps.
#
#   ./scripts/accv2026/submit_rebuttal_sweeps.sh              # all three modes
#   MODES=matched ./scripts/accv2026/submit_rebuttal_sweeps.sh
#   DRY_RUN=1 ./scripts/accv2026/submit_rebuttal_sweeps.sh    # print only
set -uo pipefail

REPO=/data/wesleyferreiramaia/infoRates
cd "${REPO}"

MODES="${MODES:-matched span covstride}"
# FineGym is absent: both its checkpoints and its source videos are gone from
# /data and /scratch, so it cannot be re-evaluated without re-downloading.
DATASETS="${DATASETS:-ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens}"
# A100 (partition gpu) for the torchvision/SlowFast checkpoints, H200
# (cenvalarc.gpu) for the Transformers and VideoMamba — same split the
# published sweeps used.
A100_MODELS="${A100_MODELS:-r3d_18 mc3_18 r2plus1d_18 slowfast_r50}"
H200_MODELS="${H200_MODELS:-timesformer vivit videomae videomamba}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p evaluations/accv2026/logs

submit() {
    local mode=$1 partition=$2 model=$3
    local name="rb-${mode}-${model}"
    # DATASETS is space-separated, and sbatch splits --export on commas, so a
    # value with spaces cannot travel inside --export. Export it into this
    # shell's environment and let --export=ALL carry it over intact.
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] MODE=${mode} MODELS=${model} DATASETS='${DATASETS}'" \
             "sbatch --partition=${partition} --time=${TIME_LIMIT}" \
             "--job-name=${name} --export=ALL slurm_rebuttal_sweep.sbatch"
        return
    fi
    echo -n "  ${name} -> "
    MODE="${mode}" MODELS="${model}" DATASETS="${DATASETS}" \
        sbatch --partition="${partition}" \
               --time="${TIME_LIMIT}" \
               --job-name="${name}" \
               --export=ALL \
               scripts/accv2026/slurm_rebuttal_sweep.sbatch
}

for mode in ${MODES}; do
    echo "=== mode=${mode} ==="
    for model in ${A100_MODELS}; do submit "${mode}" gpu "${model}"; done
    for model in ${H200_MODELS}; do submit "${mode}" cenvalarc.gpu "${model}"; done
done

echo
echo "Queue:"
squeue -u "$(whoami)" -o "%.10i %.14P %.20j %.8T %.10M" | head -30
