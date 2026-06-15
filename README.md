# InfoRates: Spatiotemporal Aliasing in Video Action Recognition

<p align="center">
  <strong>A Cross-Architecture Analysis at Scale</strong><br>
  ACCV 2026 &nbsp;|&nbsp; Mi3 Lab, UC Merced
</p>

<p align="center">
  <a href="https://mi3-inforates.streamlit.app/">Interactive Dashboard</a> &nbsp;|&nbsp;
  <a href="CONTRIBUTING_NEW_DATASETS.md">Add a Dataset</a> &nbsp;|&nbsp;
  <a href="#citation">BibTeX</a>
</p>

---

## Overview

We study how **spatial resolution**, **temporal coverage**, and **frame stride** jointly affect video recognition accuracy across 8 architectures and 8 datasets — spanning CNNs, Transformers, and State Space Models. Using 8,000+ evaluation configurations (5 resolutions × 5 coverages × 5 strides per architecture), we identify the dominant aliasing factors, rank datasets by temporal demand, and characterize architectural robustness profiles.

**Key takeaway**: Coverage is the dominant effect (ANOVA F=178.94), outweighing stride and resolution combined. Temporal aliasing severity is governed by the attention mechanism type, not the architecture family — TimeSformer degrades just 10.3pp on average, while SlowFast drops 42.1pp at stride 16.

---

## Results

### Temporal Demand Score (TDS)

TDS measures the accuracy drop from stride 1 → 16 at 100% coverage, averaged over all 8 models. It captures intrinsic dataset temporal difficulty independent of architecture choice.

| Dataset | TDS ↑ harder | Domain | Temporal Characteristic |
|---------|-------------:|--------|-------------------------|
| AUTSL | 58.3 pp | Sign language | Gestural order is semantically essential |
| FineGym | 55.9 pp | Fine-grained sport | Phase transitions critical for class separation |
| SSv2 | 29.5 pp | Causal actions | Directionality and causality collapse under coarse stride |
| DriveAct | 23.7 pp | In-vehicle activity | Short, sharp actions sensitive to frame gaps |
| Diving-48 | 21.2 pp | Fine-grained diving | Entry angle discriminates categories |
| HMDB-51 | 18.5 pp | General sports | Mixed temporal demand |
| EPIC-Kitchens | 10.9 pp | Egocentric kitchen | Object interaction, not motion, dominates |
| UCF-101 | 5.5 pp | Appearance-based | Scene context sufficient, motion mostly redundant |

### Effect of Coverage (F = 178.94, η² = 0.63–0.88)

Coverage is the strongest factor by a wide margin — 13.6× the F-statistic of resolution and 2.2× that of stride. This holds across all 8 architectures and all 8 datasets.

| Coverage | Mean Top-1 |
|---------:|-----------:|
| 10% | 21.9% |
| 25% | 27.7% |
| 50% | 39.0% |
| 75% | 48.7% |
| 100% | 51.9% |

### Effect of Stride by Architecture Family (coverage = 100%)

Temporal aliasing differs dramatically by attention mechanism, not by CNN vs. Transformer family:

| Architecture Family | stride 1 | stride 16 | Drop |
|--------------------|--------:|---------:|-----:|
| Transformer | 64.4% | 40.0% | 24.4 pp |
| CNN (3D conv) | 59.7% | 32.2% | 27.5 pp |
| SSM (VideoMamba) | 73.2% | 33.2% | 40.0 pp |
| Dual-stream (SlowFast) | 67.4% | 25.3% | 42.1 pp |

**Robustness ranking** (mean TDS across 8 datasets):

| Model | Avg. TDS ↓ | Family |
|-------|----------:|--------|
| TimeSformer | **10.3 pp** | Transformer |
| MC3-18 | 22.6 pp | CNN |
| R3D-18 | 29.7 pp | CNN |
| ViViT | 30.1 pp | Transformer |
| R2+1D | 30.0 pp | CNN |
| VideoMAE | 32.8 pp | Transformer |
| VideoMamba | 40.0 pp | SSM |
| SlowFast-R50 | 42.1 pp | Dual-CNN |

### Effect of Spatial Resolution

Evaluated via P3-retraining: each model is fine-tuned at a target resolution (10 epochs) and then evaluated at that same resolution. Without retraining, transformers degrade severely at non-native sizes; with retraining, the gap narrows but does not close.

| Family | Eval@48px | Eval@112px | Eval@224px |
|--------|----------:|-----------:|-----------:|
| **Cross-resolution (no retraining)** | | | |
| CNN | 54.8% | 59.7% | 33.6% |
| Transformer | 23.7% | 58.8% | 63.7% |
| SSM | 29.1% | 40.0% | 49.2% |
| **P3-retrained at target resolution** | | | |
| CNN | 49.3% | 67.7% | 69.2% |
| Transformer | 36.6% | 60.3% | 71.4% |
| SSM | 31.5% | 45.1% | 59.7% |

CNNs transfer reasonably across resolutions (trained at 112px, still 54.8% at 48px). Transformers are resolution-sensitive without bicubic positional embedding interpolation.

### Per-Dataset Best Model

At stride 1, coverage 100%, native resolution:

| Dataset | Best Model | Top-1 |
|---------|-----------|------:|
| UCF-101 | VideoMAE | 95.4% |
| HMDB-51 | VideoMAE | 84.0% |
| AUTSL | SlowFast-R50 | 82.3% |
| FineGym | SlowFast-R50 | 78.2% |
| DriveAct | VideoMAE | 73.9% |
| Diving-48 | ViViT | 53.0% |
| SSv2 | VideoMAE | 52.4% |
| EPIC-Kitchens | SlowFast-R50 | 39.4% |

---

## Evaluation Protocol

### Models (8 architectures)

| Model | Family | Frames | Native Res |
|-------|--------|-------:|----------:|
| R3D-18 | CNN | 16 | 112 px |
| MC3-18 | CNN | 16 | 112 px |
| R2+1D | CNN | 16 | 112 px |
| SlowFast-R50 | Dual-CNN | 32 | 224 px |
| TimeSformer | Transformer | 8 | 224 px |
| ViViT | Transformer | 32 | 224 px |
| VideoMAE | Transformer | 16 | 224 px |
| VideoMamba | SSM | 8 | 224 px |

### Datasets (8)

| Dataset | Domain | Classes | Temporal Type |
|---------|--------|--------:|---------------|
| AUTSL | Sign language | 226 | Gestural sequence |
| FineGym | Fine-grained gym | 97 | Phase transition |
| SSv2 | Causal actions | 174 | Directionality |
| DriveAct | In-vehicle | 34 | Short actions |
| Diving-48 | Fine-grained diving | 48 | Entry kinematics |
| HMDB-51 | General sports | 51 | Mixed |
| EPIC-Kitchens | Egocentric kitchen | 89 | Object interaction |
| UCF-101 | Appearance-based | 101 | Scene context |

All datasets use a stratified 20-clip-per-class validation split.

### Sweep Design

```
5 spatial resolutions  ×  5 coverage levels  ×  5 stride values
    (48, 96, 112, 160, 224 px)   (10, 25, 50, 75, 100%)   (1, 2, 4, 8, 16)
```

Total: **8,000+ evaluation configurations** (8 models × 8 datasets × 125 settings).

### P3 Retraining

For each (model, dataset, resolution) triple, we fine-tune from the ImageNet/Kinetics pretrained checkpoint for 10 epochs. Transformers and SSMs use bicubic interpolation to resize positional embeddings to the target resolution before fine-tuning, avoiding the accuracy cliff caused by discarding mismatched weights.

---

## Repository Structure

```
infoRates/
├── evaluations/accv2026/
│   ├── manifests/                        # 20-clip-per-class validation splits
│   ├── coverage_stride_sweep/            # Original 7-dataset temporal sweeps
│   ├── coverage_stride_resolution_sweep/ # Multi-resolution sweeps (FineGym + new datasets)
│   ├── p3_retrained/                     # P3-retrained checkpoint evaluations
│   └── e4_anova/                         # ANOVA results (per model × dataset)
├── dashboard/
│   ├── app.py                            # Streamlit interactive dashboard
│   └── data/
│       ├── sweep_summary.csv             # 1,425 rows: temporal sweep at native resolution
│       ├── p3_results.csv                # 324 rows: spatial resolution evaluation
│       └── retrained_spatial.csv         # 316 rows: P3-retrained checkpoint accuracies
├── scripts/accv2026/
│   ├── sweep_coverage_stride_resolution.py
│   ├── eval_p3_retrained.py
│   └── integrate_finegym_to_dashboard.py
└── src/info_rates/models/
    └── model_factory.py                  # Bicubic pos-embed interpolation for Transformers
```

---

## Interactive Dashboard

**https://mi3-inforates.streamlit.app/**

The dashboard provides:
- **Overview & TDS**: Dataset ranking by temporal demand, architecture family comparison
- **Accuracy Explorer**: Interactive (coverage, stride, resolution) sliders per model and dataset
- **Spatial Resolution Analysis**: Cross-resolution vs. P3-retrained accuracy curves
- **Aliasing Curves**: Stride degradation profiles per architecture
- **Architecture Recommender**: Guided model selection based on deployment constraints

To run locally:
```bash
cd dashboard
pip install -r ../requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

## Contributing

To add a new dataset, see [CONTRIBUTING_NEW_DATASETS.md](CONTRIBUTING_NEW_DATASETS.md). The workflow covers:
1. Preparing a 20-clip-per-class validation manifest
2. Training P3-retrained checkpoints at 5 resolutions (≈4–8 hours, all 8 models)
3. Running the coverage × stride × resolution sweep
4. Running the integration script to update dashboard CSVs

---

## Citation

```bibtex
@inproceedings{maia2026inforates,
  title     = {Spatiotemporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale},
  author    = {Maia, Wesley and others},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

---

## Related Work

- VideoMAE: Tong et al., NeurIPS 2022
- VideoMamba: Li et al., ECCV 2024
- ViViT: Arnab et al., ICCV 2021
- TimeSformer: Bertasius et al., ICML 2021
- SlowFast: Feichtenhofer et al., ICCV 2019

---

## License

MIT License — see [LICENSE](LICENSE) for details.
