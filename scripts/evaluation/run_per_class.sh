#!/bin/bash
#SBATCH --job-name=videomae_per_class
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/per_class_%j.log

cd /home/wesleyferreiramaia/data/infoRates

# Load environment
source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

# Run per-class analysis
torchrun --nproc_per_node=2 scripts/run_eval.py \
  --model fine_tuned_models/fine_tuned_videomae_ucf101 \
  --manifest data/UCF101_data/manifests/ucf101_50f.csv \
  --out data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_temporal_sampling.csv \
  --per-class \
  --per-class-out data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_per_class.csv \
  --ddp \
  --no-wandb

echo "Per-class analysis complete!"
