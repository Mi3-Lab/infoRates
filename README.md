# InfoRates — Adaptive Temporal Frame Allocation for Video Recognition

**ACCV 2026 submission** · Mi3 Lab

We study how many frames a video model actually needs. Our method predicts the minimum frame budget per video — saving compute without sacrificing accuracy — and characterizes each dataset's *temporal demand* (TDS score).

---

## Requirements

- Python 3.10+
- CUDA GPU (trained on A100 40GB and H200 141GB)
- ~512 GB scratch storage for datasets + checkpoints

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs src/info_rates as a package
```

---

## Datasets

All datasets live at `data/`. Download each and place them as shown:

| Dataset | Path | Classes | Download |
|---------|------|---------|----------|
| SSV2 | `data/Something_data/` | 174 | [20bn-something-something-v2](https://developer.qualcomm.com/software/ai-datasets/something-something) |
| UCF-101 | `data/UCF101_data/` | 101 | [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) |
| HMDB-51 | `data/HMDB51_data/` | 51 | [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |
| Diving-48 | `data/Diving48_data/` | 48 | [Diving48](http://www.svcl.ucsd.edu/projects/action_quality_assessment/) |
| EPIC-Kitchens | `data/EPIC_data/` | 97 | `python scripts/accv2026/download_epic_clips.py` |
| AUTSL | `data/AUTSL_data/` | 226 | [Kaggle: sttaseen/autsl](https://www.kaggle.com/datasets/sttaseen/autsl) |
| DriveAct | `data/DriveAct_data/` | 34 | [DriveAct](https://driveact.de/) |
| Kinetics-400 | `data/Kinetics400_data/` | 400 | [K400 val](https://github.com/cvdfoundation/kinetics-dataset) |

For AUTSL and DriveAct, run preprocessing after download:

```bash
python scripts/accv2026/preprocess_autsl.py   # generates data/AUTSL_data/splits/
python scripts/accv2026/preprocess_driveact.py # generates data/DriveAct_data/splits/
```

---

## Training

Models: **R3D-18**, **MC3-18**, **R2Plus1D-18**, **SlowFast-R50** · **TimeSformer**, **ViViT**, **VideoMAE**

### Automated (recommended)

The feeder daemon submits Slurm jobs as GPU slots open, until all datasets × models are done:

```bash
nohup bash scripts/accv2026/feeder_submit_jobs.sh &
```

It calls `submit_missing_jobs.sh` every 2 minutes — idempotent, safe to re-run.

### Manual (one dataset)

```bash
# CNN models (R3D-18, MC3-18, SlowFast-R50) — requires A100
export DATASET=hmdb51   # hmdb51 | diving48 | epic_kitchens | autsl | driveact
bash scripts/accv2026/run_a100_dataset_all_cnn.sh

# Transformers (TimeSformer, ViViT) — requires H200 or large-memory GPU
bash scripts/accv2026/run_h200_dataset_all_transformer.sh
```

Each script is idempotent: skips training if a checkpoint exists, skips eval if results exist.

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | required | dataset name (see table above) |
| `EPOCHS` | `10` | training epochs |
| `WANDB_PROJECT` | `inforates-accv2026` | W&B project |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache |

---

## Current Results (as of 2026-05-27)

Top-1 accuracy at fixed frame budgets (4 / 8 / 16 / 32 frames). `—` = training or eval pending. VideoMamba †val_acc = training val accuracy (8f); fixed-budget eval still running.

### Something-Something v2 (SSv2) — 174 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 9.8% | 19.7% | 37.1% | 36.9% |
| MC3-18 | 8.2% | 18.8% | 33.6% | 34.5% |
| R2Plus1D-18 | 12.6% | 24.3% | 42.6% | 42.1% |
| SlowFast-R50 | 6.6% | 15.2% | 33.3% | 49.5% |
| TimeSformer | 31.8% | 42.3% | 41.3% | 41.7% |
| ViViT | 8.4% | 17.5% | 30.5% | 38.3% |
| VideoMAE | 21.0% | 39.5% | 52.3% | 51.9% |
| VideoMamba† | — | — (46.8%) | — | — |

### UCF-101 — 101 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 59.5% | 72.6% | 81.2% | 81.4% |
| MC3-18 | 72.9% | 80.9% | 85.4% | 85.1% |
| R2Plus1D-18 | 70.0% | 81.6% | 88.6% | 89.0% |
| SlowFast-R50 | 50.1% | 66.2% | 81.3% | 87.6% |
| TimeSformer | 90.0% | 91.0% | 91.2% | 90.9% |
| ViViT | 75.3% | 86.9% | 92.5% | 94.3% |
| VideoMAE | 81.4% | 91.4% | 95.4% | 95.5% |
| VideoMamba | 85.0% | 88.4% | 88.2% | 87.8% |

### HMDB-51 — 51 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 49.2% | 67.1% | 80.3% | 80.1% |
| MC3-18 | 63.5% | 71.2% | 78.6% | 78.2% |
| R2Plus1D-18 | 46.2% | 63.2% | 73.1% | 74.6% |
| SlowFast-R50 | 35.1% | 44.7% | 65.1% | 79.3% |
| TimeSformer | 73.0% | 79.9% | 80.0% | 79.8% |
| ViViT | 52.4% | 66.1% | 75.4% | 80.2% |
| VideoMAE | 51.5% | 73.6% | 84.0% | 84.4% |
| VideoMamba† | — | — (70.7%) | — | — |

### DriveAct — 34 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 47.8% | 56.2% | 68.3% | 67.2% |
| MC3-18 | 55.1% | 65.8% | 69.0% | 68.5% |
| R2Plus1D-18 | 37.7% | 49.8% | 62.5% | 61.8% |
| SlowFast-R50 | 42.6% | 53.3% | 66.7% | 72.5% |
| TimeSformer | 64.7% | 67.6% | 68.8% | 66.5% |
| ViViT | 48.9% | 55.8% | 62.5% | 67.4% |
| VideoMAE | 40.2% | 56.0% | 74.1% | 72.5% |
| VideoMamba† | — | — (69.5%) | — | — |

### Diving-48 — 48 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 5.9% | 14.4% | 28.8% | 28.8% |
| MC3-18 | 8.2% | 19.5% | 31.6% | 33.4% |
| R2Plus1D-18 | 8.6% | 16.8% | 35.3% | 34.7% |
| SlowFast-R50 | 5.8% | 14.5% | 26.4% | 50.5% |
| TimeSformer | 23.6% | 38.0% | 36.9% | 38.0% |
| ViViT | 7.9% | 19.9% | 35.1% | 53.0% |
| VideoMAE | 8.6% | 27.6% | 48.6% | 49.9% |
| VideoMamba† | — | — (43.6%) | — | — |

### AUTSL — 226 classes (sign language)

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 4.7% | 24.5% | 75.0% | 74.4% |
| MC3-18 | 4.1% | 37.5% | 63.7% | 63.7% |
| R2Plus1D-18 | 8.4% | 30.2% | 75.9% | 75.0% |
| SlowFast-R50 | 1.6% | 12.7% | 41.8% | 82.3% |
| TimeSformer | 52.0% | 66.8% | 66.2% | 67.0% |
| ViViT | 8.4% | 25.5% | 61.2% | 74.6% |
| VideoMAE | 17.7% | 43.2% | 79.5% | 78.9% |
| VideoMamba† | — | — | — | — |

### EPIC-Kitchens — 97 classes — parcial (R3D-18/MC3-18 reeval, VideoMAE retreinando, VideoMamba treinando)

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | — | — | — | — |
| MC3-18 | — | — | — | — |
| R2Plus1D-18 | 18.7% | 30.9% | 52.0% | 51.4% |
| SlowFast-R50 | 8.0% | 15.5% | 30.1% | 43.0% |
| TimeSformer | 22.5% | 37.4% | 37.4% | 36.1% |
| ViViT | 11.7% | 23.2% | 36.2% | 40.2% |
| VideoMAE | — | — | — | — |
| VideoMamba† | — | — (52.2%) | — | — |

---

## Analysis Pipeline

After all training jobs finish:

```bash
bash scripts/accv2026/run_post_completion_analyses.sh
```

Or run steps individually (numbered scripts = pipeline order):

| Script | What it does |
|--------|-------------|
| `02_make_manifests.py` | Build per-class eval manifests |
| `04_compute_temporal_demand.py` | Compute TDS score per dataset |
| `05_compute_temporal_metrics.py` | AUC and critical budget per model |
| `07_dataset_temporal_demand.py` | Dataset-level TDS summary |
| `08_compile_paper_results.py` | Paper tables |
| `09_plot_paper_figures.py` | Paper figures (Fig 1–9) |
| `10_per_class_temporal_analysis.py` | Per-class temporal analysis |
| `06_fde_adaptive_routing.py` | FDE routing evaluation |
| `11_spectral_router.py` | Spectral router |
| `12_confidence_cascade.py` | Confidence cascade analysis |
| `13_knapsack_confidence.py` | Knapsack frame allocator |
| `14_plot_routing_comparison.py` | Routing comparison figures |
| `15_baseline_comparison.py` | Multi-dataset baseline table |

Results are written to `evaluations/accv2026/paper_results/`.

---

## Repository Structure

```
.
├── src/info_rates/               # Python package
│   ├── data/datasets.py          # unified loader: load_dataset(name, root)
│   ├── metrics/                  # TDS, AUC, temporal metrics
│   └── models/                   # model factory
├── scripts/accv2026/             # active pipeline
│   ├── 00–15_*.py                # numbered analysis steps
│   ├── train_torchvision.py      # R3D-18, MC3-18 trainer
│   ├── train_slowfast.py         # SlowFast-R50 trainer
│   ├── train_transformers.py     # TimeSformer, ViViT trainer
│   ├── eval_fixed_budget.py      # fixed-budget evaluation
│   ├── feeder_submit_jobs.sh     # automation daemon
│   ├── submit_missing_jobs.sh    # idempotent job submission
│   ├── run_a100_dataset_all_cnn.sh
│   ├── run_h200_dataset_all_transformer.sh
│   ├── preprocess_autsl.py
│   └── preprocess_driveact.py
├── scripts/legacy/               # archived experiments (ECCV, SSV2-only)
├── evaluations/accv2026/
│   ├── fixed_budget/             # per-model per-dataset eval results
│   ├── paper_results/            # final tables and figures
│   ├── manifests/                # eval manifests
│   └── logs/                     # Slurm job logs
├── data/                         # datasets (symlink to /scratch on cluster)
├── fine_tuned_models/            # checkpoints (symlink to /scratch on cluster)
└── requirements.txt
```

---

## Cluster Notes (Mi3 Lab HPC)

- **A100 partition:** `gpu` — CNN models (max 4 concurrent jobs)
- **H200 partition:** `cenvalarc.gpu` — transformers (max 4 concurrent jobs)
- `data/` and `fine_tuned_models/` are symlinked to `/scratch` (512 GB)
- `HF_HOME=/scratch/wesleyferreiramaia/hf_unified` — consolidated model cache
