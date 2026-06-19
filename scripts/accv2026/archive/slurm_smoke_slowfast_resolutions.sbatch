#!/usr/bin/env bash
#SBATCH --job-name=smoke-sf-res
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=disabled

echo "=== SlowFast smoke test (AdaptiveAvgPool fix) — todas as 5 resoluções ==="
nvidia-smi | head -4

for res in 96 112 160 224 336; do
    echo ""
    echo "--- Testing SlowFast @ ${res}px ---"
    result=$(python scripts/accv2026/train_slowfast.py \
        --dataset ssv2 --epochs 1 --batch-size 4 \
        --input-size $res --max-train-samples 8 --max-val-samples 4 \
        --no-wandb --save-path /tmp/smoke_sf_${res}px \
        2>&1 | grep -E "Epoch|Training complete|RuntimeError|Error|size" | tail -3)
    echo "$result"
    if echo "$result" | grep -q "Training complete"; then
        echo "  ✅ SlowFast @ ${res}px PASSED"
    else
        echo "  ❌ SlowFast @ ${res}px FAILED"
    fi
done

echo ""
echo "=== SlowFast smoke DONE ==="
