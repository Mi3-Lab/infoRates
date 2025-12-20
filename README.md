# TemporalProbe: Temporal Sampling for Action Recognition

This repository investigates how temporal sampling (frame coverage and stride) affects action recognition performance in modern video models, providing a practical framework to optimize spatiotemporal resolution and computational efficiency.

## Overview
- Benchmark attention-based video architectures (e.g., TimeSformer).
- Fixed-frame clip generation; vary coverage (%) and stride.
- Quantify accuracy vs. temporal density and compute cost.
- Config-driven training and evaluation with DDP support.
- WandB integration for experiment tracking.

## Getting Started

### 1) Setup Python Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare Data
Extract UCF-101 and split files into `UCF101_data/`:
```bash
ls UCF101_data/UCF-101/           # 101 action classes
ls UCF101_data/ucfTrainTestlist/  # train/testlist files
```

### 3) Build Fixed-Frame Clips (50f)
```bash
python scripts/build_clips.py \
    --video-root UCF101_data/UCF-101 \
    --out UCF101_data/UCF101_50f \
    --frames 50
```

### 4) Build Manifests
```bash
# Train manifest
python scripts/build_manifest_from_clips.py \
    --clips-dir UCF101_data/UCF101_50f \
    --split-file UCF101_data/ucfTrainTestlist/trainlist01.txt \
    --out UCF101_data/manifests/ucf101_50f_train.csv

# Dev/Test manifest
python scripts/build_manifest_from_clips.py \
    --clips-dir UCF101_data/UCF101_50f \
    --split-file UCF101_data/ucfTrainTestlist/testlist01.txt \
    --out UCF101_data/manifests/ucf101_50f_dev.csv
```

### 5) Training (DDP on 2 GPUs)
Edit `config.yaml` with your hyperparameters, then:
```bash
salloc --partition=gpu --nodes=1 --gres=gpu:2 --constraint=a100 --time=8:00:00 --pty bash
cd /path/to/repo
source .venv/bin/activate
bash scripts/train_ddp.sh
```

### 6) Evaluation (DDP)
Evaluation runs temporal sampling experiments with coverage/stride sweeps:
```bash
bash scripts/pipeline_eval.sh config.yaml
```
Or standalone:
```bash
python scripts/run_eval.py --config config.yaml
```

## Configuration
All settings are in `config.yaml`:
- **Train settings**: `train_epochs`, `train_batch_size`, `train_learning_rate`, etc.
- **Eval settings**: `eval_manifest`, `eval_coverages`, `eval_strides`, `eval_sample_size`, etc.
- **Data**: `clips_dir`, `train_manifest`, `results_dir`
- **Flags**: `use_dev_split`, `build_manifest_if_missing`

## Directory Structure
```
.
├── config.yaml                 # Main config file (training/eval settings)
├── scripts/
│   ├── build_clips.py         # Create fixed-frame clips from videos
│   ├── build_manifest_from_clips.py  # Build manifests from existing clips
│   ├── train_timesformer.py   # Training script (DDP-enabled)
│   ├── train_ddp.sh           # DDP training launcher
│   ├── run_eval.py            # Evaluation script (temporal sampling sweep)
│   └── pipeline_eval.sh       # End-to-end eval pipeline
├── UCF101_data/
│   ├── UCF-101/               # Videos (101 classes)
│   ├── ucfTrainTestlist/      # Official train/test splits
│   ├── UCF101_50f/            # Fixed-frame clips
│   ├── manifests/             # CSV manifests for train/dev/test
│   └── results/               # Evaluation results
├── models/                    # Saved model checkpoints
└── notebooks/
    └── infoRatesUCF.ipynb     # Original analysis notebook
```

## Notes
- GPU/CUDA required for training and evaluation.
- WandB logging enabled by default; set `train_disable_wandb: true` to disable.
- See [UCF101_data/README.md](UCF101_data/README.md) for dataset-specific notes.
- DDP-ready: use `torchrun --nproc_per_node=N` or `NPROC_PER_NODE=N bash scripts/train_ddp.sh`.

## Citation
If you use this work, please cite:
```bibtex
@software{temporalprobe2025,
  author = {Ferreira Maia, Wesley},
  title = {TemporalProbe: Temporal Sampling for Action Recognition},
  year = {2025},
  url = {https://github.com/yourusername/temporal-probe}
}
```
