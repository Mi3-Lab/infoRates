#!/usr/bin/env bash
#SBATCH --job-name=accv26-vmamba-smoke
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

echo "=== VideoMamba smoke test — job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | $(date)"
nvidia-smi | head -6 || true

# Verify mamba-ssm is installed
source .venv_mamba/bin/activate
pip install opencv-python-headless --quiet 2>/dev/null || true
python3 -c "
import causal_conv1d, mamba_ssm
print('causal_conv1d:', causal_conv1d.__version__)
print('mamba_ssm:', mamba_ssm.__version__)
from mamba_ssm.modules.mamba_simple import Mamba
import torch
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 4, 64).cuda()
y = m(x)
print('Mamba forward OK:', x.shape, '->', y.shape)
"

export PYTHONPATH=src
export WANDB_MODE=offline  # offline for smoke test
export ACCV_JOB_ID="${SLURM_JOB_ID}"

echo ""
echo "=== Training 1 epoch on SSv2 (200 samples) ==="
python scripts/accv2026/train_videomamba.py \
    --dataset ssv2 \
    --epochs 1 \
    --batch-size 8 \
    --num-workers 2 \
    --max-train-samples 200 \
    --max-val-samples 100 \
    --save-path fine_tuned_models/videomamba_smoke_ssv2 \
    --no-wandb

echo ""
echo "=== Smoke forward pass test ==="
python3 -c "
import sys; sys.path.insert(0, 'src')
sys.path.insert(0, 'third_party/videomamba_repo/videomamba/video_sm')
from info_rates.models.videomamba_model import load_videomamba_checkpoint
import torch
model, processor, meta = load_videomamba_checkpoint('fine_tuned_models/videomamba_smoke_ssv2')
model = model.cuda().eval()
import numpy as np
frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
inputs = processor(frames, return_tensors='pt')
with torch.no_grad():
    out = model(pixel_values=inputs['pixel_values'].cuda())
print('Checkpoint load OK | logits shape:', out.logits.shape)
print('num_labels:', meta['num_labels'])
"

echo "=== VideoMamba smoke test COMPLETE ==="
