# Temporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale

<div align="center">

**ACCV 2026** · Mi3 Lab · University of California, Merced

[Dashboard](https://inforates.streamlit.app) · [Paper (draft)](#)

</div>

---

> We present the first large-scale, cross-architecture study of temporal aliasing in video action recognition, evaluating 8 architectures — 3D CNNs, dual-pathway, Transformers, and SSMs — across 7 diverse datasets (1,400 evaluation configurations). Our principal finding is that **aliasing behavior is governed by attention mechanism type, not architecture family**: VideoMamba (SSM) and TimeSformer (divided attention) alias 3–5× less than CNNs and ViViT (factorized attention), despite ViViT being a Transformer. We introduce the **Temporal Demand Score (TDS)**, a dataset-level metric that ranks datasets by temporal frequency demand consistently across all 8 models. Cross-resolution evaluation further shows the same architectural split in the spatial domain. An entropy-based routing method then exploits this to route 77% of videos to 4-frame inference with no accuracy loss.

---

## Key Findings

### 1 — Temporal aliasing follows attention type, not architecture family

Mean accuracy drop (stride=1 → stride=16, 100% coverage):

| Model | Family | Mean drop | AUTSL | SSv2 | UCF-101 |
|-------|--------|:---------:|------:|-----:|--------:|
| **VideoMamba** | SSM | **8pp** | †collapse | +14pp | +0pp |
| **TimeSformer** | Transformer (divided) | **8pp** | +16pp | +13pp | +0pp |
| MC3-18 | CNN | 21pp | +56pp | +27pp | +3pp |
| R2+1D | CNN | 28pp | +67pp | +31pp | +5pp |
| R3D-18 | CNN | 28pp | +68pp | +28pp | +6pp |
| VideoMAE | Transformer (MAE) | 32pp | +61pp | +34pp | +4pp |
| **ViViT** | Transformer (factorized) | 34pp | +62pp | +31pp | +5pp |
| SlowFast | Dual-CNN | **42pp** | +78pp | +43pp | +15pp |

†VideoMamba/AUTSL: feature collapse due to K400→sign language domain gap (0.4% at all strides). Excluded from TDS computation.

**ViViT aliases as badly as CNNs despite being a Transformer** — factorized space-time attention decouples temporal reasoning, making the time dimension vulnerable to frame dropout.

### 2 — Temporal Demand Score (TDS) is architecture-independent

TDS ranks datasets by mean accuracy drop (stride=1→16), consistent across all 8 models (Spearman ρ=0.97):

| AUTSL | SSv2 | DriveAct | Diving-48 | HMDB-51 | EPIC-Kitchens | UCF-101 |
|------:|-----:|---------:|----------:|--------:|--------------:|--------:|
| 58.3pp | 27.6pp | 21.9pp | 19.2pp | 16.6pp | 9.7pp | 4.9pp |

### 3 — Spatial robustness follows the same split (SSv2, no retraining)

| Model | 96px | 112px | 160px | 224px | 336px | Δ_max |
|-------|-----:|------:|------:|------:|------:|------:|
| R3D-18 (native 112px) | 35.9 | **37.1** | 30.3 | 17.2 | 5.8 | −31pp |
| R2+1D (native 112px) | 40.1 | **42.6** | 36.2 | 20.8 | 5.9 | −37pp |
| MC3-18 (native 112px) | 32.4 | **33.6** | 27.8 | 14.8 | 4.4 | −29pp |
| SlowFast (native 224px) | 27.7 | 32.4 | 45.7 | **49.5** | 36.0 | −22pp |
| TimeSformer (native 224px) | 39.3 | 39.4 | 41.4 | **42.3** | 42.1 | −3pp |
| ViViT (native 224px) | 36.4 | 37.1 | 38.1 | **38.3** | 38.0 | −2pp |
| VideoMAE (native 224px) | 48.9 | 49.2 | 51.5 | **52.3** | 51.9 | −3pp |
| VideoMamba (native 224px) | 37.5 | 39.3 | 42.5 | **43.9** | 43.8 | −6pp |

CNNs degrade severely off-resolution; Transformers and VideoMamba stay within ±6pp across the full 96–336px range.

### 4 — Entropy routing: 77% of videos routed cheaply

Zero-training routing at two operating points (4-frame cheap, 16-frame dense) routes based on prediction confidence:

| Method | Avg frames | SSv2 Top-1 | Backbone |
|--------|:----------:|:----------:|----------|
| FrameExit | 9.9 | 38.4% | TimeSformer |
| Fixed 8f | 8.0 | 42.3% | TimeSformer |
| **Entropy (ours)** | **7.7** | **42.5%** | TimeSformer |
| Oracle upper bound | 7.7 | 47.3% | TimeSformer |

On average across all 8 models and 7 datasets, **77% of videos are routed to 4-frame inference** with no accuracy penalty.

---

## Interactive Dashboard

Results are available at **[inforates.streamlit.app](https://inforates.streamlit.app)**:
- Coverage × stride heatmaps for all 8 models × 7 datasets
- TDS ranking and spectral validation
- ANOVA effect sizes and Levene variance inflation
- Entropy routing curves and compute trade-offs
- Spatial resolution robustness (SSv2)
- Architecture Recommender (RAG + Groq hybrid)

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

**Requirements:** Python 3.10+, CUDA GPU (tested on A100 40GB and H200 141GB NVL).

---

## Datasets

| Dataset | Classes | Domain | TDS |
|---------|--------:|--------|----:|
| AUTSL | 226 | Sign language | 58.3pp |
| SSv2 | 174 | Causal/temporal | 27.6pp |
| DriveAct | 33 | In-vehicle | 21.9pp |
| Diving-48 | 48 | Fine-grained | 19.2pp |
| HMDB-51 | 51 | Sports | 16.6pp |
| EPIC-Kitchens | 89 | Egocentric | 9.7pp |
| UCF-101 | 101 | Appearance | 4.9pp |

---

## Citation

```bibtex
@inproceedings{maia2026temporal,
  title     = {Temporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale},
  author    = {Maia, Wesley and others},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

---

## License

MIT License. VideoMamba weights from [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba) (MIT).
