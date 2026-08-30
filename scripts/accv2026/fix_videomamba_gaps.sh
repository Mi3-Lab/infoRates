#!/usr/bin/env bash
# Re-run the VideoMamba pairs that failed in the first pass, once per mode.
#
# diving48     get_checkpoint returned a retired checkpoint (accv_meta.json
#              renamed to .bak after it collapsed at 6.8% val_acc). It now
#              validates loadability and falls through to full_e10_h200.
# kinetics400  the K400 pretrained weights live on /scratch, but the path was
#              resolved against the repo root on /data. Now searched in both.
#
# Both fixes are in place; this only fills the gaps they left.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
for DS in diving48 kinetics400; do
  for MODE in matched span antialias covstride; do
    out=evaluations/accv2026/rebuttal_sweeps/$MODE/videomamba_$DS/summary.csv
    [ -f "$out" ] && { echo "[skip] $MODE/$DS ja existe"; continue; }
    until MODE=$MODE MODELS=videomamba DATASETS=$DS \
          sbatch --partition=gpu,cenvalarc.gpu --time=06:00:00 \
                 --job-name="rb-$MODE-vm-$DS" --export=ALL \
                 scripts/accv2026/slurm_rebuttal_sweep.sbatch 2>/dev/null | grep -q Submitted; do
      sleep 120
    done
    echo "[$(date +%H:%M)] submetido $MODE/videomamba/$DS"
  done
done
echo "[$(date +%H:%M)] todas as lacunas submetidas"
