#!/usr/bin/env bash
#SBATCH --job-name=smoke-vmamba-res
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:30:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv_mamba/bin/activate
export PYTHONPATH=src
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=disabled

echo "=== VideoMamba smoke test (WITH pos_embed interpolation fix) ==="
nvidia-smi | head -4

for res in 96 112 160 224 336; do
    echo ""
    echo "--- Testing VideoMamba @ ${res}px ---"
    result=$(python scripts/accv2026/train_videomamba.py \
        --dataset ssv2 --epochs 1 --batch-size 2 \
        --input-size $res --max-train-samples 4 --max-val-samples 2 \
        --no-wandb --save-path /tmp/smoke_vmamba_fix_${res}px \
        2>&1 | grep -E "pos_embed|Spatial|Epoch|Training complete|RuntimeError|Error" | tail -5)
    echo "$result"
    if echo "$result" | grep -q "Training complete"; then
        echo "  ✅ VideoMamba @ ${res}px PASSED"
    else
        echo "  ❌ VideoMamba @ ${res}px FAILED"
    fi
done

echo ""
echo "=== VideoMamba smoke DONE ==="
