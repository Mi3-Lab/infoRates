#!/usr/bin/env bash
# Submit one VideoMamba training job per dataset in parallel on H200 GPUs.
# Safe to re-run: skips datasets already in queue OR with existing checkpoints.
# If the QOS job limit is hit, re-run once GPU slots open.
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates

DATASETS="${DATASETS:-ssv2 ucf101 hmdb51 driveact diving48 autsl epic_kitchens}"
SBATCH="scripts/accv2026/slurm_h200_videomamba_dataset.sbatch"
EPOCHS="${EPOCHS:-10}"

# Get set of job names currently in queue
queued_names=$(squeue -u "${USER}" --format="%j" 2>/dev/null | tail -n +2)

submitted=0
skipped=0
failed=0

for ds in $DATASETS; do
    CHECKPOINT="fine_tuned_models/accv2026_videomamba_${ds}_full_e${EPOCHS}_h200"
    if [[ -f "${CHECKPOINT}/config.json" ]]; then
        echo "  SKIP (checkpoint exists): DATASET=${ds}"
        ((skipped++)) || true
        continue
    fi
    if echo "${queued_names}" | grep -qx "${ds}"; then
        echo "  SKIP (already in queue): DATASET=${ds}"
        ((skipped++)) || true
        continue
    fi
    result=$(sbatch --job-name="${ds}" --export=ALL,DATASET="${ds}" "${SBATCH}" 2>&1)
    if echo "${result}" | grep -q "Submitted batch job"; then
        jobid=$(echo "${result}" | awk '{print $NF}')
        echo "  Submitted: DATASET=${ds}  jobid=${jobid}"
        # Refresh queue list after submission
        queued_names=$(squeue -u "${USER}" --format="%j" 2>/dev/null | tail -n +2)
        ((submitted++)) || true
    else
        echo "  FAILED to submit DATASET=${ds}: ${result}" >&2
        ((failed++)) || true
    fi
done

echo ""
echo "Summary: ${submitted} submitted, ${skipped} skipped (done/queued), ${failed} failed (QOS limit)"
[[ $failed -gt 0 ]] && echo "Re-run this script once GPU slots open to submit remaining datasets."
echo "Monitor with: squeue -u ${USER}"
