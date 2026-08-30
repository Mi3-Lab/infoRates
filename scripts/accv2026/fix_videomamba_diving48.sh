#!/usr/bin/env bash
# Re-run the one (model, dataset) pair that failed: videomamba/diving48.
#
# get_checkpoint used to return accv2026_videomamba_diving48_224px_e10_h200,
# whose accv_meta.json had been renamed to .bak to retire it (its val_acc is
# 6.8%, a collapsed run). The directory still existed, so resolution succeeded
# and the failure surfaced only inside the loader. get_checkpoint now validates
# loadability and falls through to full_e10_h200; this re-runs the gap.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
for MODE in matched span antialias covstride; do
  until MODE=$MODE MODELS=videomamba DATASETS=diving48 \
        sbatch --partition=gpu,cenvalarc.gpu --time=06:00:00 \
               --job-name="rb-$MODE-vm-div48" --export=ALL \
               scripts/accv2026/slurm_rebuttal_sweep.sbatch 2>/dev/null | grep -q "Submitted"; do
    sleep 120
  done
  echo "[$(date +%H:%M)] submetido $MODE/videomamba/diving48"
done
echo "[$(date +%H:%M)] todos os modos submetidos"
