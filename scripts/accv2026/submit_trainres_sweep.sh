#!/usr/bin/env bash
# Submit coverage×stride sweep jobs for all resolution-retrained checkpoints.
# Empirical QOS limit: 6 total jobs (pending+running). Leaves 1 slot free.
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_trainres_sweep.sbatch
LOCKS=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks/trainres_sweep
MAX_JOBS=7        # gpu(4 max) + cenvalarc.gpu(4 max) = 8; leave 1 free → 7
USER=wesleyferreiramaia

mkdir -p "$LOCKS"

MODELS="r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba"
DATASETS="ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens"
RESOLUTIONS="96 112 160 224"

running() { squeue -u "$USER" -h -r | wc -l; }

done_count=0
skip_count=0
submit_count=0

for model in $MODELS; do
    for ds in $DATASETS; do
        for res in $RESOLUTIONS; do
            lock="$LOCKS/${model}_${ds}_${res}.lock"
            # Check if sweep_summary.csv already exists (already done)
            out_dir="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/coverage_stride_sweep/${model}_${ds}_trainres${res}"
            if [[ -f "${out_dir}/sweep_summary.csv" ]]; then
                ((skip_count++)) || true
                continue
            fi
            [[ -f "$lock" ]] && continue   # Already submitted this session

            # Wait for a free slot
            while [[ $(running) -ge $MAX_JOBS ]]; do
                sleep 30
            done

            # Try both partitions; skip on QOS error without stopping daemon
            echo "[$(date +%H:%M:%S)] Submitting ${model}/${ds}@${res}px ..."
            jid=$(MODEL=$model DATASET=$ds TRAIN_RES=$res \
                  sbatch --parsable --partition=gpu,cenvalarc.gpu "$SBATCH" 2>&1) || {
                echo "  [WARN] sbatch failed: $jid — will retry next pass"
                rm -f "$lock"   # allow retry
                sleep 60
                continue
            }
            touch "$lock"
            echo "  job $jid"
            ((submit_count++)) || true
            sleep 2
        done
    done
done

echo ""
echo "Done. Submitted: $submit_count | Skipped (already done): $skip_count"
