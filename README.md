# InfoRates — Spatiotemporal Aliasing in Video Action Recognition

<p align="center">
  <em>A Cross-Architecture Analysis at Scale</em><br><br>
  <strong>ACCV 2026 · Mi3 Lab, UC Merced</strong>
</p>

<p align="center">
  <a href="https://mi3-inforates.streamlit.app/"><img src="https://img.shields.io/badge/Dashboard-Live-brightgreen?logo=streamlit" alt="Dashboard"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Status-Submitted-orange" alt="Status">
</p>

---

## Overview

We study how **spatial resolution**, **temporal coverage**, and **frame stride** jointly affect video recognition accuracy across **8 architectures** and **8 datasets**, spanning CNNs, Transformers, and State Space Models. Using **8,000+ evaluation configurations** (5 resolutions × 5 coverages × 5 strides per architecture), we identify the dominant aliasing factors, rank datasets by temporal demand, and characterize architectural robustness profiles.

**Key takeaways:**
- **Coverage dominates** (ANOVA F=178.94, η²=0.63–0.88) — 13.6× the effect of resolution and 2.2× stride
- **Attention type, not architecture family, governs aliasing robustness** — TimeSformer degrades just 10.3 pp vs. SlowFast's 42.1 pp at stride 16
- **Dataset temporal demand is stable** — Spearman ρ=0.97 across 8 architectures; CNN-only / Transformer-only / Transformer+SSM sub-pools all reproduce the same ranking (ρ ≥ 0.976)
- **Spectral validation** — higher inter-frame optical flow frequency correlates negatively with stride sensitivity (Spearman ρ=−0.549, p=0.0006, n=35 across 7 datasets × 5 resolutions)

---

## Table of Contents

1. [Results](#results)
2. [Models & Datasets](#models--datasets)
3. [Sweep Design](#sweep-design)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Running the Dashboard](#running-the-dashboard)
7. [Contributing](#contributing)
8. [Citation](#citation)

---

## Results

### Temporal Demand Score (TDS)

TDS measures the accuracy drop from stride 1 → stride 16 at 100% coverage, averaged over all 8 models. It ranks datasets by their intrinsic temporal difficulty independent of architecture choice. The ranking is stable under architecture-family ablation and bootstrap resampling (95% CI for adjacent-rank gaps shown in supplementary).

| Dataset | TDS ↑ harder | Domain | Temporal Characteristic |
|---------|-------------:|--------|-------------------------|
| AUTSL | 58.3 pp | Sign language | Gestural order is semantically essential |
| FineGym | 58.1 pp | Fine-grained sport | Phase transitions critical for class separation |
| SSv2 | 27.6 pp | Causal actions | Directionality and causality collapse under coarse stride |
| DriveAct | 21.9 pp | In-vehicle activity | Short, sharp actions sensitive to frame gaps |
| Diving-48 | 19.2 pp | Fine-grained diving | Entry angle discriminates categories |
| HMDB-51 | 16.6 pp | General sports | Mixed temporal demand |
| EPIC-Kitchens | 9.7 pp | Egocentric kitchen | Object interaction, not motion, dominates |
| UCF-101 | 4.9 pp | Appearance-based | Scene context sufficient, motion mostly redundant |

### ANOVA Effect Sizes (Coverage is King)

| Factor | F-statistic | η² range | Relative to Coverage |
|--------|------------:|----------:|---------------------|
| **Coverage** | **178.94** | **0.63–0.88** | — |
| Stride | 82.1 | 0.08–0.35 | 2.2× weaker |
| Resolution | 13.2 | 0.04–0.18 | 13.6× weaker |

Coverage accuracy plateau (all 8 datasets × 8 models averaged):

| Coverage | Mean Top-1 |
|---------:|-----------:|
| 10% | 21.9% |
| 25% | 27.7% |
| 50% | 39.0% |
| 75% | 48.7% |
| 100% | 51.9% |

### Stride Degradation by Architecture (coverage = 100%)

| Architecture | Stride 1 | Stride 16 | Drop | Family |
|---|---:|---:|---:|---|
| TimeSformer | 62.3% | 52.0% | **10.3 pp** | Transformer |
| MC3-18 | 54.1% | 31.5% | 22.6 pp | CNN |
| R3D-18 | 54.1% | 24.4% | 29.7 pp | CNN |
| R2+1D | 56.2% | 26.2% | 30.0 pp | CNN |
| ViViT | 58.2% | 28.1% | 30.1 pp | Transformer |
| VideoMAE | 65.4% | 32.6% | 32.8 pp | Transformer |
| VideoMamba | 73.2% | 33.2% | 40.0 pp | SSM |
| SlowFast-R50 | 67.4% | 25.3% | **42.1 pp** | Dual-CNN |

### Spatial Resolution Robustness

Native-resolution checkpoints evaluated at other input sizes (no retraining). CNN backbones transfer well; Transformers require bicubic positional-embedding interpolation to avoid accuracy collapse.

| Family | @48 px | @96 px | @112 px | @224 px |
|--------|-------:|-------:|--------:|--------:|
| CNN | 48.2% | 58.0% | 59.7% | 33.6% |
| Transformer | 10.4% | 57.7% | 58.8% | 63.7% |
| SSM | 6.3% | 37.7% | 40.0% | 49.2% |

### Best Model per Dataset

At stride 1, coverage 100%, native resolution:

| Dataset | Best Model | Top-1 |
|---------|-----------|------:|
| UCF-101 | VideoMAE | 95.4% |
| HMDB-51 | VideoMAE | 84.0% |
| AUTSL | SlowFast-R50 | 82.3% |
| FineGym | SlowFast-R50 | 79.9% |
| DriveAct | VideoMAE | 73.9% |
| Diving-48 | ViViT | 53.0% |
| SSv2 | VideoMAE | 52.4% |
| EPIC-Kitchens | SlowFast-R50 | 39.4% |

---

## Models & Datasets

### Models (8 architectures)

| Model | Family | Frames | Native Res | Notes |
|-------|--------|-------:|----------:|-------|
| R3D-18 | CNN | 16 | 112 px | 3D ResNet baseline |
| MC3-18 | CNN | 16 | 112 px | Mixed convolutions |
| R(2+1)D-18 | CNN | 16 | 112 px | Factored spatiotemporal |
| SlowFast-R50 | Dual-CNN | 32 | 224 px | Fast + slow pathways |
| TimeSformer | Transformer | 8 | 224 px | Divided space-time attention |
| ViViT | Transformer | 32 | 224 px | Tubelet factored attention |
| VideoMAE | Transformer | 16 | 224 px | Masked autoencoder pre-training |
| VideoMamba | SSM | 8 | 224 px | Bidirectional Mamba |

### Datasets (8)

| Dataset | Domain | Classes | Clips/class | Temporal Type |
|---------|--------|--------:|------------:|---------------|
| AUTSL | Sign language | 226 | 20 | Gestural sequence |
| FineGym | Fine-grained gym | 97 | 20 | Phase transition |
| SSv2 | Causal actions | 174 | 20 | Directionality |
| DriveAct | In-vehicle | 34 | 20 | Short actions |
| Diving-48 | Fine-grained diving | 48 | 20 | Entry kinematics |
| HMDB-51 | General sports | 51 | 20 | Mixed |
| EPIC-Kitchens | Egocentric kitchen | 89 | 20 | Object interaction |
| UCF-101 | Appearance-based | 101 | 20 | Scene context |

All datasets use a stratified 20-clip-per-class validation split (see `evaluations/accv2026/manifests/`).

---

## Sweep Design

```
5 spatial resolutions  ×  5 coverage levels  ×  5 stride values
   48 / 96 / 112 / 160 / 224 px     10 / 25 / 50 / 75 / 100 %     1 / 2 / 4 / 8 / 16
```

**Total: 8,000+ evaluation configurations** — 8 models × 8 datasets × 125 settings.

Each configuration is evaluated using the model's native-resolution checkpoint. For the **spatial resolution experiment (P3)**, models are also fine-tuned from scratch at each target resolution (10 epochs, bicubic PE interpolation for Transformers/SSMs).

---

## Repository Structure

```
infoRates/
├── paper/                            # ACCV 2026 manuscript
│   ├── main.tex                      # Main paper (9 pages)
│   ├── supplementary.tex             # Supplementary (S1–S12)
│   ├── main.bib                      # Bibliography
│   └── images/                       # Figures used in paper
│
├── dashboard/                        # Streamlit interactive dashboard
│   ├── app.py                        # Main app (~1,400 lines)
│   ├── data/                         # Pre-aggregated CSVs for fast loading
│   │   ├── sweep_summary.csv         # 1,600 rows: full temporal sweep
│   │   ├── p3_results.csv            # 308 rows: cross-resolution eval (no retrain)
│   │   ├── retrained_spatial.csv     # 316 rows: P3-retrained accuracy
│   │   └── anova_results.csv         # Per-model/dataset ANOVA η²
│   └── requirements.txt
│
├── evaluations/accv2026/
│   ├── manifests/                    # 20-clip-per-class validation splits
│   ├── coverage_stride_sweep/        # Per-model/dataset sweep results (native res)
│   ├── coverage_stride_resolution_sweep/ # Multi-resolution sweep results
│   ├── spatial_resolution_sweep/     # Cross-resolution eval (no retrain)
│   ├── e3_spectral/                  # Optical flow spectral analysis data
│   ├── e4_anova/                     # ANOVA outputs per model × dataset
│   └── paper_figures/                # Generated paper figures
│
├── scripts/accv2026/
│   ├── 00_prepare_datasets.py        # Dataset pipeline (00–19)
│   │   …
│   ├── train_*.py                    # Training scripts per model family
│   ├── sweep_coverage_stride.py      # Temporal sweep runner
│   ├── sweep_spatial_resolution.py   # Spatial sweep runner
│   ├── analyze_tds_robustness.py     # Family ablation + bootstrap CI
│   ├── analyze_nyquist_spectral_v2.py # Spectral validation (n=35)
│   ├── slurm_*.sbatch                # Slurm job templates
│   ├── submit_*.sh                   # Daemon launchers
│   └── archive/                      # Completed one-off scripts (kept for reference)
│
├── src/info_rates/                   # Python package (pip install -e .)
│   ├── models/                       # model_factory.py, architecture wrappers
│   ├── training/                     # DDP training utilities
│   ├── sampling/                     # Coverage/stride samplers
│   └── data/                         # Dataset loaders and manifests
│
├── docs/                             # Reference documentation
│   ├── ACCV_2026_ARCHITECTURE_AND_SAMPLING_PROTOCOL.md
│   ├── ACCV_2026_DATASET_PREP.md
│   ├── ACCV_2026_H200_RUNBOOK.md
│   └── DATA_ANOMALY_ANALYSIS.md
│
├── assets/                           # Teaser figures for README/website
├── pyproject.toml                    # Package metadata
├── requirements.txt                  # Training/eval dependencies
├── CONTRIBUTING_NEW_DATASETS.md      # How to add a new dataset
├── PROGRESS.md                       # Live experiment tracker
└── CITATION.cff                      # Preferred citation
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ (A100 or H200 recommended for training)
- ~200 GB storage for all 8 datasets

### Setup

```bash
git clone <repo-url>
cd infoRates

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the info_rates package + all training/eval dependencies
pip install -e ".[dashboard]"
```

For VideoMamba (requires custom CUDA kernels):
```bash
python -m venv .venv_mamba
source .venv_mamba/bin/activate
pip install -e ".[dashboard]"
# See docs/ACCV_2026_ARCHITECTURE_AND_SAMPLING_PROTOCOL.md for fake-nvcc workaround
```

---

## Running the Dashboard

**Live:** [https://mi3-inforates.streamlit.app/](https://mi3-inforates.streamlit.app/)

To run locally:
```bash
cd dashboard
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The dashboard requires only `dashboard/data/*.csv` — no GPU, no model checkpoints.

Pages:
- **Overview & TDS** — Dataset ranking, architecture family comparison
- **Accuracy Explorer** — Interactive (coverage, stride, resolution) sweeps per model/dataset
- **Spatial Resolution** — Cross-resolution accuracy curves (no-retrain vs. P3-retrained)
- **Aliasing Curves** — Stride degradation profiles per architecture
- **Architecture Recommender** — Guided model selection by deployment constraints

---

## Contributing

We welcome contributions of new **datasets** and **architectures**. See [CONTRIBUTING_NEW_DATASETS.md](CONTRIBUTING_NEW_DATASETS.md) for the complete protocol, including:

- Preparing a stratified validation manifest (20 clips/class)
- Running the coverage × stride × resolution sweep (~1,000 configs per architecture)
- P3 retraining protocol and checkpoint naming conventions
- Integrating results into the dashboard CSVs and ANOVA outputs

To contribute, open a pull request or contact **wesleymaia999@gmail.com**.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{maia2026inforates,
  title     = {Spatiotemporal Aliasing in Video Action Recognition:
               A Cross-Architecture Analysis at Scale},
  author    = {Maia, Wesley and Greer, Ross},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026},
  note      = {\url{https://mi3-inforates.streamlit.app/}}
}
```

Or use the [CITATION.cff](CITATION.cff) file (supported by GitHub, Zenodo, and Zotero).

---

## License

MIT License — see [paper/LICENSE](paper/LICENSE) for details.

---

## Acknowledgements

- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) — Tong et al., NeurIPS 2022
- [VideoMamba](https://github.com/OpenGVLab/VideoMamba) — Li et al., ECCV 2024
- [ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) — Arnab et al., ICCV 2021
- [TimeSformer](https://github.com/facebookresearch/TimeSformer) — Bertasius et al., ICML 2021
- [SlowFast](https://github.com/facebookresearch/SlowFast) — Feichtenhofer et al., ICCV 2019
- Compute provided by UC Merced HPC (A100 + H200 nodes)
