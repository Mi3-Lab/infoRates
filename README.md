# InfoRates: Adaptive Temporal Frame Allocation for Video Recognition

<div align="center">

**ACCV 2026** · Mi3 Lab · University of São Paulo

[Paper](#) · [arXiv](#) · [BibTeX](#citation)

</div>

---

> Every action class has a *temporal frequency* — sample below it and you get aliasing: different actions become indistinguishable. We provide the first systematic characterization of temporal aliasing frequencies across 7 diverse datasets and 8 architectures (CNNs, Transformers, and SSMs), introducing the **Temporal Demand Score (TDS)** as a principled, model-independent measure of a dataset's Nyquist rate. We further show that **aliasing behavior is architecture-dependent**: VideoMamba (SSM) aliases 4–16× less than CNNs on the same dataset, while TimeSformer's divided attention is 4× more robust than ViViT's factorized attention.

<p align="center">
  <img src="assets/fig9_main_comparison.png" width="85%" alt="InfoRates vs fixed-budget baseline across 7 datasets"/>
</p>

---

## Method

InfoRates characterizes each dataset's **Temporal Demand Score (TDS)** — how much accuracy is lost when reducing the frame budget from 32 to 4 frames. Using TDS and per-video temporal signals, we route each video to the minimum frame budget that preserves accuracy.

<p align="center">
  <img src="assets/fig3_tds_dataset_ranking.png" width="70%" alt="TDS ranking: AUTSL demands 60pp more accuracy from more frames; UCF-101 only 16pp"/>
</p>

**Key findings:**

**Finding 1 — Temporal aliasing is architecture-dependent** (stride 1→16, 100% coverage on SSv2/AUTSL):

| Model | Family | AUTSL loss | Diving-48 loss | SSv2 loss |
|-------|--------|:----------:|:--------------:|:---------:|
| VideoMamba | SSM | †collapse | +5pp | +14pp |
| TimeSformer | Transformer | +16pp | +2pp | +13pp |
| MC3-18 | CNN | +56pp | +12pp | +27pp |
| R3D-18 | CNN | +68pp | +16pp | +28pp |
| R2Plus1D | CNN | +67pp | +19pp | +31pp |
| ViViT | Transformer | +62pp | +36pp | +31pp |
| SlowFast | CNN-dual | **+78pp** | +40pp | +43pp |

† VideoMamba AUTSL = **feature collapse** (K400→sign language domain gap): 0.4% at ALL strides. This is not temporal robustness — the model never learned AUTSL.

On valid datasets (SSv2, HMDB, Diving, DriveAct, EPIC): **VideoMamba (avg 8pp) and TimeSformer (avg 9pp) are equally robust** — both 3–5× more robust than CNNs (avg 24–46pp) and ViViT (avg 34pp). The key contrast: **ViViT aliases as badly as CNNs despite being a Transformer** — factorized space-time attention decouples temporal reasoning, making the time dimension vulnerable to sparse sampling.

**Finding 2 — TDS ranks are consistent across architectures**: AUTSL (sign language, +60pp) > Diving-48 (+29pp) > SSv2 (+26pp) > UCF-101 (+16pp)

**Finding 3 — Spatial aliasing is also architecture-dependent** (E6, SSv2 at 5 resolutions):

| Model | 96px | 112px | 160px | 224px | 336px |
|-------|-----:|------:|------:|------:|------:|
| R3D-18 (native 112px) | — | **37.1%** | 30.3% | 17.2% | — |
| R2Plus1D (native 112px) | 40.1% | **42.6%** | 36.2% | 20.8% | 5.9% |
| VideoMAE (native 224px) | 48.9% | 49.2% | 51.5% | **52.3%** | 51.9% |
| ViViT (native 224px) | 36.4% | 37.1% | 38.1% | **38.3%** | 38.0% |
| VideoMamba (native 224px) | 37.5% | 39.3% | 42.5% | **43.9%** | 43.8% |

CNNs alias sharply above their native resolution; Transformers and SSMs are spatially robust across the tested range.

**Finding 4 — VideoMAE aliases temporally like a CNN (+32pp avg) despite being a Transformer:**
MAE pre-training teaches spatial reconstruction (robust to spatial masking) but NOT temporal robustness — skipping frames is different from masking patches. VideoMAE clips at stride 4→8 just like CNNs.

**Finding 5 — Statistical validation (E2/E4):**
- **ANOVA η²**: Stride explains 7–35% of accuracy variance (large for CNNs, small for SSM/TSF). Coverage explains 52–90%.
- **Levene's test**: Stride significantly increases inter-class std up to 2.0× (p<0.001) — aliasing is not just mean drop, it increases class disparity.

**Finding 6 — Action taxonomy (E5):**
UCF-101 "Low" sensitivity classes: −0.3pp at stride=16 (completely static actions). AUTSL "Low" tier still loses 38pp (entire dataset is high-frequency). Quantifies the within-dataset heterogeneity.

---

## Experiment Status

| Experiment | Status | Key finding |
|-----------|--------|------------|
| **E1** Coverage×Stride (8 models × 7 datasets × 25 configs) | ✅ 1400/1400 | Full aliasing curves; VideoMAE +32pp surprises |
| **E2** Variance / Levene | ✅ Complete | Stride increases inter-class std up to 2.0× |
| **E3** Spectral (optical flow ↔ aliasing) | 🔄 Running | Nyquist validation via Pearson r |
| **E4** ANOVA η² effect sizes | ✅ Complete | Coverage dominates; stride η²=0.08 (SSM) vs 0.35 (SlowFast) |
| **E5** Action sensitivity taxonomy | ✅ Complete | High/Moderate/Low tiers for all 7 datasets |
| **E6** Spatial resolution sweep | ✅ Complete (SSv2) | VideoMAE flat 96–336px; CNNs brittle OOD |
| **P3** Resolution retraining | 🔄 14% (33/224) | SlowFast@96px > @224px: SSv2 +8.9pp |
| **E7** Adaptive routing (C3) | ❌ Next | Closes method contribution |

---

## Results

<p align="center">
  <img src="assets/fig1_ssv2_budget_curves.png" width="80%" alt="Accuracy vs frame budget for all 8 models on SSv2"/>
</p>

Fixed-budget Top-1 accuracy across 7 datasets and 8 models (4 / 8 / 16 / 32 frames):

<details>
<summary><b>Something-Something v2 (174 classes)</b></summary>

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 9.8% | 19.7% | 37.1% | 36.9% |
| MC3-18 | 8.2% | 18.8% | 33.6% | 34.5% |
| R2Plus1D-18 | 12.6% | 24.3% | 42.6% | 42.1% |
| SlowFast-R50 | 6.6% | 15.2% | 33.3% | 49.5% |
| TimeSformer | 31.8% | 42.3% | 41.3% | 41.7% |
| ViViT | 8.4% | 17.5% | 30.5% | 38.3% |
| VideoMAE | 21.0% | 39.5% | **52.3%** | 51.9% |
| VideoMamba | 31.8% | **43.9%** | 44.4% | 44.2% |

</details>

<details>
<summary><b>UCF-101 (101 classes)</b></summary>

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 59.5% | 72.6% | 81.2% | 81.4% |
| MC3-18 | 72.9% | 80.9% | 85.4% | 85.1% |
| R2Plus1D-18 | 70.0% | 81.6% | 88.6% | 89.0% |
| SlowFast-R50 | 50.1% | 66.2% | 81.3% | 87.6% |
| TimeSformer | 90.0% | 91.0% | 91.2% | 90.9% |
| ViViT | 75.3% | 86.9% | 92.5% | 94.3% |
| VideoMAE | 81.4% | 91.4% | 95.4% | **95.5%** |
| VideoMamba | 85.0% | **88.4%** | 88.2% | 87.8% |

</details>

<details>
<summary><b>HMDB-51 · DriveAct · Diving-48 · AUTSL · EPIC-Kitchens</b></summary>

| Model | HMDB-51 @16f | DriveAct @16f | Diving-48 @32f | AUTSL @16f | EPIC @16f |
|-------|-------------:|--------------:|---------------:|-----------:|----------:|
| R3D-18 | 80.3% | 68.3% | 28.8% | 75.0% | 37.2% |
| MC3-18 | 78.6% | 69.0% | 33.4% | 63.7% | 36.2% |
| R2Plus1D-18 | 73.1% | 62.5% | 34.7% | 75.9% | 35.5% |
| SlowFast-R50 | 79.3% | 72.5% | **50.5%** | **82.3%** | 39.4% |
| TimeSformer | 80.0% | 68.8% | 38.0% | 67.0% | 31.5% |
| ViViT | 80.2% | 67.4% | 53.0% | 74.6% | 32.9% |
| VideoMAE | **84.4%** | **74.1%** | 49.9% | 79.5% | **37.7%** |
| VideoMamba | 69.7% | 58.0% | 31.4% | 0.4%† | 28.4% |

† VideoMamba does not converge on AUTSL: K400 backbone produces near-identical features for sign language clips (feature collapse). All other models learn normally.

</details>

**Temporal Demand Score by dataset** (mean accuracy gain 4→32 frames, all models):

| AUTSL | Diving-48 | SSv2 | HMDB-51 | EPIC | DriveAct | UCF-101 |
|------:|----------:|-----:|--------:|-----:|---------:|--------:|
| +59.9pp | +29.1pp | +26.1pp | +24.2pp | +20.6pp | +18.2pp | +15.9pp |

---

## Installation

```bash
# Standard models (CNNs + Transformers)
git clone https://github.com/mi3lab/infoRates
cd infoRates
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# VideoMamba (SSM) — requires separate environment
python -m venv .venv_mamba && source .venv_mamba/bin/activate
pip install -r requirements_mamba.txt
pip install -e .
```

**Requirements:** Python 3.10+, CUDA GPU (tested on A100 40GB and H200 141GB), ~512 GB storage.

---

## Datasets

| Dataset | Path | Classes | Source |
|---------|------|---------|--------|
| SSV2 | `data/Something_data/` | 174 | [Qualcomm AI](https://developer.qualcomm.com/software/ai-datasets/something-something) |
| UCF-101 | `data/UCF101_data/` | 101 | [UCF](https://www.crcv.ucf.edu/data/UCF101.php) |
| HMDB-51 | `data/HMDB51_data/` | 51 | [Brown](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |
| Diving-48 | `data/Diving48_data/` | 48 | [SVCL](http://www.svcl.ucsd.edu/projects/action_quality_assessment/) |
| EPIC-Kitchens | `data/EPIC_data/` | 97 | [EPIC-Kitchens-100](https://epic-kitchens.github.io/2022) |
| AUTSL | `data/AUTSL_data/` | 226 | [Kaggle](https://www.kaggle.com/datasets/sttaseen/autsl) |
| DriveAct | `data/DriveAct_data/` | 34 | [driveact.de](https://driveact.de/) |

```bash
# Preprocessing (AUTSL and DriveAct only)
python scripts/accv2026/preprocess_autsl.py
python scripts/accv2026/preprocess_driveact.py

# EPIC-Kitchens clip extraction
python scripts/accv2026/download_epic_clips.py
```

---

## Training

All scripts are idempotent — they skip training if a checkpoint exists and skip evaluation if results exist.

```bash
# CNN models (R3D-18, MC3-18, R2Plus1D-18, SlowFast-R50) — A100
export DATASET=ssv2   # ssv2 | ucf101 | hmdb51 | diving48 | epic_kitchens | autsl | driveact
bash scripts/accv2026/run_a100_dataset_all_cnn.sh

# Transformer models (TimeSformer, ViViT, VideoMAE) — H200
bash scripts/accv2026/run_h200_dataset_all_transformer.sh

# VideoMamba (SSM) — H200, requires .venv_mamba
DATASET=ssv2 bash scripts/accv2026/run_h200_multidata_videomamba.sh
```

---

## Evaluation

**Experiment status (31 May 2026):**

| Experiment | Status | Progress |
|-----------|--------|----------|
| E1: Temporal aliasing (coverage×stride sweep) | 🔄 83% | 1162/1400 configs · UCF-101 fix deployed |
| E6: Spatial resolution sweep (5 pts per model) | 🔄 87% | 7/8 models done on SSv2 |
| P3: Retraining at 5 resolutions | 🔄 20% | 44/224 checkpoints |

```bash
# W&B live results
# https://wandb.ai/mi3lab/inforates-accv2026

# Monitor job queue
squeue -u $USER --format="%.10i %.22j %.8T %P"

# Coverage×Stride sweep for 1 model+dataset
python scripts/accv2026/sweep_coverage_stride.py --model r3d_18 --dataset ssv2

# Spatial resolution sweep for 1 model+dataset
python scripts/accv2026/sweep_spatial_resolution.py --model timesformer --dataset ssv2
```

Figures are saved as **PDF** (for LaTeX) and **PNG at 300 DPI** (for review/slides) in `evaluations/accv2026/paper_results/figures/`.

---

## Citation

```bibtex
@inproceedings{maia2026inforates,
  title     = {InfoRates: Adaptive Temporal Frame Allocation for Video Recognition},
  author    = {Maia, Wesley and others},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

VideoMamba backbone weights are from [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba) (MIT License).
