#!/usr/bin/env bash
# Submit the VideoMamba per-clip sweeps that did not fit under the QOS submit
# limit, retrying until each one is accepted.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
submit() {  # dataset [train_res]
  local ds=$1 tr=${2:-}
  [ -f evaluations/accv2026/coverage_stride_sweep_perclip/videomamba_$ds/cov100_s16_samples.csv ] && { echo "[skip] $ds"; return; }
  squeue -u "$USER" -h -o %j | grep -qx "vm-pc-$ds" && { echo "[queued] $ds"; return; }
  until DATASET=$ds TRAIN_RES=$tr sbatch --partition=gpu,cenvalarc.gpu --job-name=vm-pc-$ds \
        --export=ALL scripts/fg2027/slurm_videomamba_perclip.sbatch 2>/dev/null | grep -q Submitted; do
    sleep 300
  done
  echo "[$(date '+%F %H:%M')] submitted $ds"
}
submit driveact
submit epic_kitchens
submit autsl 224
echo "[$(date '+%F %H:%M')] all submitted"
