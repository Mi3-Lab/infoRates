# Spectral Analysis (Single Guide)

This document consolidates all spectral analysis documentation into one place, including the tables you asked for.

## Overview

We provide a quantitative validation of the Nyquist-Shannon sampling principle for video action recognition. Action dynamics are treated as temporal signals derived from optical flow. High-frequency actions require dense sampling; low-frequency actions tolerate aggressive subsampling.

Key result (demo data): dominant frequency correlates strongly with aliasing sensitivity (Spearman rho ~ 0.94, p < 0.01).

## Theory (Short)

Nyquist-Shannon: a signal with maximum frequency f_max must be sampled at rate f_s >= 2 * f_max to avoid aliasing. In videos, optical-flow-derived motion energy provides the temporal signal. If sampling stride pushes the effective sampling rate below the critical frequency, aliasing degrades recognition.

## What We Compute

1. Optical flow (dense Farneback, optional sparse LK).
2. Temporal aggregation (mean flow magnitude per frame).
3. Power spectral density (Welch method).
4. Spectral descriptors per class:
   - Dominant frequency (peak of PSD)
   - Spectral centroid (power-weighted mean frequency)
   - Low-frequency energy ratio (1-5 Hz band)
   - Spectral flatness (tonality)
5. Correlation with empirical aliasing sensitivity (accuracy drop from 100% to 25% coverage).

## Tables (Key Results)

### Correlation Summary (Demo Results)

| Metric | Pearson r | Spearman rho | p-value | Interpretation |
| --- | --- | --- | --- | --- |
| Dominant Frequency vs Mean Drop | 0.991 | 0.943 | 0.0048 | Strong positive correlation |
| Spectral Centroid vs Mean Drop | 0.991 | 0.943 | 0.0048 | Strong positive correlation |
| Low-Freq Energy vs Mean Drop | -0.995 | -0.943 | 0.0048 | Strong negative correlation |
| Spectral Flatness vs Mean Drop | -0.982 | -0.943 | 0.0048 | Negative correlation |

### Per-Class Spectral Profile Example

| Class | Dominant Freq (Hz) | Spectral Centroid (Hz) | Low-Freq Energy Ratio | Mean Drop (%) |
| --- | --- | --- | --- | --- |
| YoYo | 6.5 | 5.8 | 0.35 | -51.4 |
| JumpingJack | 5.8 | 5.2 | 0.40 | -47.0 |
| SalsaSpin | 5.2 | 4.8 | 0.45 | -43.5 |
| Typing | 2.0 | 2.2 | 0.78 | -0.4 |
| Bowling | 1.5 | 1.8 | 0.82 | 0.4 |
| Billiards | 1.2 | 1.5 | 0.85 | -0.7 |

## Quick Start

Demo (no videos, synthetic data):

```bash
source .venv/bin/activate
python scripts/run_spectral_analysis.py --output-dir evaluations/spectral_demo
```

Real analysis (with dataset + per-class CSV):

```bash
python scripts/run_spectral_analysis.py \
  --sensitivity-csv evaluations/ucf101/vivit/vivit_per_class.csv \
  --dataset-root data/UCF101_data/UCF-101 \
  --output-dir evaluations/spectral_ucf101 \
  --max-videos-per-class 20 \
  --optical-flow-method farneback \
  --fft-method welch
```

## Outputs

The run produces:

- correlation_analysis.json
- spectral_validation_summary.csv
- 01_spectral_profiles.png
- 02_correlation_scatter.png
- 03_sensitivity_tiers_spectral.png

Images are also copied to the repository root under images/ for LaTeX inclusion.

## How To Use The Figures In LaTeX

```latex
\includegraphics{images/01_spectral_profiles.png}
\includegraphics{images/02_correlation_scatter.png}
\includegraphics{images/03_sensitivity_tiers_spectral.png}
```

## Expected Results

- High-frequency actions: dominant frequency around 5-6 Hz, large accuracy drops.
- Low-frequency actions: dominant frequency around 1-2 Hz, small or near-zero drops.
- Strong positive correlation between dominant frequency and sensitivity.
- Strong negative correlation between low-frequency energy ratio and sensitivity.

## Troubleshooting

- Missing cv2: `pip install opencv-python`
- Missing decord: `pip install decord`
- Missing pandas/scipy/matplotlib/seaborn: install them in .venv
- Class names mismatch: dataset folder names must match per-class CSV
- Slow run: reduce --max-videos-per-class, increase --subsample, or use LK flow

## For The Paper (Text Snippet)

"Spectral analysis of optical-flow dynamics provides a quantitative validation of Nyquist-Shannon sampling for action recognition. Classes with higher dominant motion frequencies exhibit larger accuracy drops under temporal undersampling, while low-frequency classes remain robust. This establishes a mechanistic link between motion frequency content and aliasing sensitivity, and supports adaptive, content-aware temporal sampling."

## Core Files

- src/info_rates/analysis/spectral_analysis.py
- scripts/run_spectral_analysis.py
- scripts/analysis/spectral_correlation.py
- scripts/verify_spectral_setup.py
