#!/usr/bin/env bash
# Submit VideoMamba fixed-budget evals for completed-training datasets.
# Safe to re-run: skips datasets that already have evals or no checkpoint.
# Handles QOS limits gracefully — continues even if sbatch is rejected.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SUBMITTED=0
SKIPPED=0
FAILED=0

for DATASET in hmdb51 driveact diving48 epic_kitchens; do
    CKPT="fine_tuned_models/accv2026_videomamba_${DATASET}_full_e10_h200"
    OUT_DIR="evaluations/accv2026/fixed_budget/videomamba_${DATASET}_full_e10_h200"
    if [[ ! -d "${CKPT}" ]]; then
        echo "[SKIP] ${DATASET}: no checkpoint at ${CKPT}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    SUMMARY=$(ls "${OUT_DIR}/"*"_fixed_budget_summary.csv" 2>/dev/null | head -1 || true)
    if [[ -n "${SUMMARY}" ]]; then
        echo "[SKIP] ${DATASET}: eval already exists at ${OUT_DIR}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    echo "[SUBMIT] ${DATASET} eval"
    if DATASET=${DATASET} sbatch --job-name="vmamba-${DATASET}-eval" \
        scripts/accv2026/slurm_h200_videomamba_eval.sbatch; then
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "[WARN] ${DATASET}: sbatch rejected (QOS limit?); run again when slot opens"
        FAILED=$((FAILED + 1))
    fi
done
echo "Done. submitted=${SUBMITTED} skipped=${SKIPPED} failed=${FAILED}"
