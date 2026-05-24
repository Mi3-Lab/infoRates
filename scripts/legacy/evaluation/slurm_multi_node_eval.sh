#!/bin/bash
#SBATCH --job-name=kinetics400_eval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Carregar módulos necessários (ajuste conforme seu cluster)
module load cuda/11.8
module load python/3.12

# Ativar ambiente virtual
source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

# Configurar variáveis de ambiente para PyTorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12345
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "Node rank: $NODE_RANK"

# Executar avaliação
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/run_eval.py \
    --config config.yaml \
    --dataset kinetics400 \
    --model-path facebook/timesformer-base-finetuned-k400 \
    --sample-size 0 \
    --batch-size 38 \
    --workers 4 \
    --ddp \
    --per-class \
    --per-class-sample-size -1 \
    --jitter-coverage-pct 0.0 \
    --wandb-project inforates-kinetics400