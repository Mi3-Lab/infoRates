#!/usr/bin/env bash
# Acceptance test for the VideoMamba/Diving-48 checkpoint.
#
# get_checkpoint resolves this pair to accv2026_videomamba_diving48_224px_e10_h200,
# whose accv_meta.json has been renamed to .bak -- deliberately, since its val_acc
# is 6.8%, i.e. a collapsed run. The published sweep reports 36.5% at cov100/s1,
# so it cannot have used that checkpoint. Rather than infer which one it did use,
# evaluate the candidates: `matched` at k=8 samples 8 frames uniformly over the
# clip, which is what cov100/s1 does, so the right checkpoint should reproduce
# 36.5% within ~1pp.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
CK=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
for c in full_e10_h200 224px_e10_v3_h200; do
  until MODE=matched MODELS=videomamba DATASETS=diving48 \
        EXTRA_ARGS="--checkpoint $CK/accv2026_videomamba_diving48_$c --tag probe_$c" \
        sbatch --partition=gpu,cenvalarc.gpu --time=01:00:00 \
               --job-name="vm-probe" --export=ALL \
               scripts/accv2026/slurm_rebuttal_sweep.sbatch 2>/dev/null | grep -q "Submitted"; do
    sleep 120
  done
  echo "[$(date +%H:%M)] submetido probe $c"
done
