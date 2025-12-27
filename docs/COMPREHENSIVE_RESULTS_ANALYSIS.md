# Understanding Aliasing of Human Activity to Optimize Spatiotemporal Resolution and Computational Efficiency in Recognition Tasks

## Abstract

This comprehensive study investigates the impact of temporal sampling strategies on human action recognition across modern video architectures, with direct implications for resource-constrained applications in healthcare, robotics, autonomous systems, and smart environments. Leveraging signal processing principles and the Nyquist-Shannon sampling theorem, we systematically characterize the critical temporal frequencies of human actions and quantify the effects of undersampling (aliasing) and oversampling (computational costs) across three state-of-the-art architectures: TimeSformer, VideoMAE, and ViViT.

Our empirical analysis across 31,023 videos from UCF-101 and Kinetics-400 datasets reveals heterogeneous temporal requirements across action categories, with high-frequency motions (e.g., diving, weightlifting, gymnastics) exhibiting extreme vulnerability to temporal undersampling while low-frequency activities (e.g., massage, dancing, skating) maintain robust performance even at aggressive sampling reductions. Statistical analysis confirms that temporal coverage accounts for ~6.1–20.6% of variance in recognition accuracy ($p<0.001$), with architecture-specific responses to stride parameters.

Key findings include: (1) identification of critical sampling thresholds below which aliasing dominates performance, (2) empirical validation of Nyquist-Shannon principles in video classification, (3) architecture-specific temporal processing characteristics, and (4) practical guidelines for optimizing spatiotemporal resolution in real-world systems. The results provide systematic frameworks for designing efficient human action recognition systems, enabling principled decisions about sensor selection, computational trade-offs, and temporal parameter optimization for applications requiring real-time, resource-constrained recognition.

**Keywords**: temporal aliasing, action recognition, Nyquist-Shannon sampling, video architectures, computational efficiency, spatiotemporal optimization

---

## 1. Methodology

### 1.1 Datasets

We evaluate temporal sampling effects across two benchmark datasets representing diverse action recognition scenarios:

**UCF-101** [8]: 13,320 videos across 101 action categories, focusing on realistic human actions with complex temporal dynamics. Test split: 12,227 videos.

**Kinetics-400** [9]: 400 action classes with 19,796 validation videos, representing diverse human activities with varying temporal characteristics.

### 1.2 Architectures

We evaluate three state-of-the-art video architectures representing different design paradigms:

**TimeSformer** [1]: Transformer-based architecture using factorized spatiotemporal attention. Input: 8 frames, pre-trained on Kinetics-400.

**VideoMAE** [2]: Masked autoencoder approach with transformer backbone. Input: 16 frames, pre-trained on Kinetics-400.

**ViViT** [3]: Vision transformer adapted for video through factorized space-time attention. Input: 32 frames, pre-trained on Kinetics-400.

All models were fine-tuned on respective training splits and evaluated on test/validation sets.

### 1.3 Temporal Sampling Protocol

We systematically explore temporal sampling effects through 25 configurations combining:

- **Coverage Levels**: 10%, 25%, 50%, 75%, 100% of available frames
- **Stride Values**: 1, 2, 4, 8, 16 frames between sampled positions

This design enables independent assessment of coverage (temporal extent) and stride (temporal density) effects.

### 1.4 Evaluation Metrics

**Primary Metrics**:
- Top-1 classification accuracy across configurations
- Per-class accuracy analysis for heterogeneity assessment
- Inference latency measurements for efficiency evaluation

**Statistical Analysis**:
- One-way ANOVA for factor significance testing
- Post-hoc pairwise comparisons with Bonferroni correction
- Effect size calculations (η², Cohen's d)
- Variance homogeneity tests (Levene's test)

### 1.5 Computing Infrastructure and Reproducibility

All training and distributed evaluation experiments were executed on the host `metalhogod` using 2× NVIDIA A100 40GB GPUs with PyTorch Distributed Data Parallel (DDP). Reproducibility artifacts (per-class CSVs, pairwise JSONs, and scripts) are available in `evaluations/` and `scripts/`.

---

## 2. Results

### 2.1 Comprehensive Performance Analysis

**Table 1: Complete Performance Results Across Datasets and Architectures**

| Dataset | Architecture | Peak Accuracy (best cfg) | Mean @100% (±std) | Best @100% (stride, acc) | Mean @75% (±std) | Mean @50% (±std) | Mean @25% (±std) | Mean @10% (±std) | Aliasing Drop (100→25) | Latency (s) | Memory (GB) |
|---------|--------------|-------------------------|-------------------:|---------------------------|-------------------:|-------------------:|-------------------:|-------------------:|----------------------:|-------------:|------------:|
| UCF-101 | TimeSformer | 85.09% (100%, stride-2) | 84.22% ± 1.19% | stride-2 (85.09%) | 84.08% ± 1.07% | 82.16% ± 1.77% | 77.36% ± 4.60% | 73.58% ± 3.92% | -6.86% | 0.017 | 2.1 |
| UCF-101 | VideoMAE | 86.90% (100%, stride-1) | 79.85% ± 6.74% | stride-1 (86.90%) | 74.92% ± 10.18% | 73.24% ± 9.04% | 62.68% ± 12.61% | 54.33% ± 11.35% | -17.18% | 0.029 | 3.2 |
| UCF-101 | ViViT | 85.49% (100%, stride-1) | 83.59% ± 2.01% | stride-1 (85.49%) | 81.49% ± 3.81% | 78.74% ± 5.47% | 70.41% ± 8.64% | 64.05% ± 7.17% | -13.18% | 0.41 | 1.8 |
| Kinetics-400 | TimeSformer | 74.19% (100%, stride-4) | 74.01% ± 0.15% | stride-4 (74.19%) | 73.29% ± 0.07% | 70.59% ± 0.39% | 63.41% ± 1.82% | 56.13% ± 2.67% | -10.60% | 0.017 | 2.1 |
| Kinetics-400 | VideoMAE | 76.52% (50%, stride-2) | 75.69% ± 0.26% | stride-1 (75.98%) | 74.78% ± 3.56% | 74.59% ± 3.56% | 68.53% ± 5.52% | 60.69% ± 4.42% | -7.16% | 0.029 | 3.2 |
| Kinetics-400 | ViViT | 76.19% (100%, stride-1) | 76.02% ± 0.27% | stride-1 (76.19%) | 74.86% ± 1.61% | 73.10% ± 1.73% | 67.78% ± 1.53% | 60.81% ± 2.25% | -8.23% | 0.41 | 1.8 |


> **Note:** "Peak" is the single best coverage×stride configuration found across all experiments; "Mean @X%" reports the mean ± std across strides at that coverage level.

**Figure 1: Coverage Degradation Patterns Across All Architectures and Datasets**

**UCF-101 — TimeSformer**

![UCF-101 TimeSformer Coverage](../evaluations/ucf101/timesformer/accuracy_vs_coverage.png)

**Description (UCF-101 — TimeSformer):** Maintains high accuracy at full coverage (~0.85) with a moderate decline as coverage decreases; curves for different strides largely overlap, indicating low sensitivity to stride and effective global temporal aggregation through attention.

**UCF-101 — VideoMAE**

![UCF-101 VideoMAE Coverage](../evaluations/ucf101/videomae/accuracy_vs_coverage.png)

**Description (UCF-101 — VideoMAE):** Shows a strong peak for stride-1 at full coverage but high volatility across strides and steep drops at low coverage (large average loss), reflecting reliance on dense temporal context and reconstruction-based representations.

**UCF-101 — ViViT**

![UCF-101 ViViT Coverage](../evaluations/ucf101/vivit/accuracy_vs_coverage.png)

**Description (UCF-101 — ViViT):** Exhibits intermediate behavior: strong performance at high coverage with greater separation between stride curves than TimeSformer; this indicates moderate sensitivity to temporal density, particularly for high-frequency classes.

**Kinetics-400 — TimeSformer**

![Kinetics-400 TimeSformer Coverage](../evaluations/kinetics400/timesformer/accuracy_vs_coverage.png)

**Description (Kinetics-400 — TimeSformer):** The optimal stride shifts toward larger strides (e.g., stride-4) when ample coverage is available; nonetheless, reducing coverage causes consistent accuracy drops—global attention helps but cannot recover missing temporal information.

**Kinetics-400 — VideoMAE**

![Kinetics-400 VideoMAE Coverage](../evaluations/kinetics400/videomae/accuracy_vs_coverage.png)

**Description (Kinetics-400 — VideoMAE):** Stable at intermediate coverages but sensitive to sparse sampling in high-frequency classes; the curves suggest VideoMAE leverages temporal correlations effectively when they are present.

**Kinetics-400 — ViViT**

![Kinetics-400 ViViT Coverage](../evaluations/kinetics400/vivit/accuracy_vs_coverage.png)

**Description (Kinetics-400 — ViViT):** Shows consistent behavior with less inter-stride variance than VideoMAE; moderate declines at reduced coverage indicate robustness for low-to-mid frequency actions.

**Figure 2: Stride-Accuracy Heatmaps** — See Supplementary Figures S1–S6 in the Supplementary Material at the end of this document.

### 2.2 Statistical Analysis of Temporal Effects

This section summarizes the inferential statistics for coverage and stride across datasets and architectures — the full ANOVA tables, pairwise coverage comparisons, and variance analyses are detailed in Section 2.4 (Statistical Hypothesis Testing) and Table 2.


### 2.3 Per-Class Heterogeneity Analysis

**Table 3: Action Categories by Aliasing Sensitivity**

| Sensitivity Tier | Δ Range | UCF-101 Classes | Kinetics-400 Classes | Representative Actions | Motion Characteristics |
|------------------|---------|-----------------|---------------------|----------------------|----------------------|
| High-Sensitivity | Δ > 25% | 12 | 15 | SalsaSpin, ThrowDiscus, YoYo | High-velocity, explosive, complex motions |
| Moderate-Sensitivity | 10% < Δ ≤ 25% | 38 | 42 | Sports, tool use, manipulation | Dynamic controlled motions |
| Low-Sensitivity | Δ ≤ 10% | 51 | 44 | Personal care, locomotion | Gentle, rhythmic, predictable motions |

**Figure 3: Per-Class Representative Trajectories**

![UCF-101 TimeSformer Classes](../evaluations/ucf101/timesformer/per_class_representative.png)
![UCF-101 VideoMAE Classes](../evaluations/ucf101/videomae/per_class_representative.png)
![UCF-101 ViViT Classes](../evaluations/ucf101/vivit/per_class_representative.png)
![Kinetics-400 TimeSformer Classes](../evaluations/kinetics400/timesformer/per_class_representative.png)
![Kinetics-400 VideoMAE Classes](../evaluations/kinetics400/videomae/per_class_representative.png)
![Kinetics-400 ViViT Classes](../evaluations/kinetics400/vivit/per_class_representative.png)

**Figure 4: Per-Class Distribution by Coverage**

![Per-Class Distributions (Composite)](../evaluations/comparative/per_class_distribution_composite.png)

**Figure X (Composite):** Per-class accuracy distributions at stride=8 across coverage levels, arranged as a 2×3 panel (rows: UCF-101, Kinetics-400; columns: TimeSformer, VideoMAE, ViViT). Medians are annotated above each box for clarity; individual point labels have been removed for readability. If you prefer separate per-model panels, high-resolution per-model images are available in each model directory (e.g., `evaluations/ucf101/timesformer/per_class_distribution_by_coverage.png`).

![Coverage-Stride Interactions](../evaluations/comparative/coverage_stride_interactions.png)

**Figure X: Coverage–Stride Interactions.** Mean accuracy (color) across coverage (rows) and stride (columns) for each dataset × model (UCF-101, Kinetics-400 × TimeSformer, VideoMAE, ViViT). Each cell displays mean accuracy for that configuration; white text indicates poorer performance while darker colors indicate higher accuracy.

**Key Patterns**:
- At full coverage: Architecture-specific optimal strides (TimeSformer: stride-8, VideoMAE/ViViT: stride-1)
- At reduced coverage: Dense sampling (stride-1) provides robustness across architectures
- Sparse sampling amplifies aliasing effects at low coverage levels

## 2.4 Statistical Hypothesis Testing

### 2.4.1 Overview and Data Sources
All reported inferential statistics below are computed from per-class accuracy vectors stored in the evaluation outputs (see `evaluations/*/*_per_class.csv`) and the precomputed summary statistics in `evaluations/*/*/statistical_results.json`. Pairwise coverage comparisons were computed using Welch's t-tests on per-class accuracies (stride = 1) with Bonferroni correction for 10 comparisons.

### 2.4.2 Comprehensive ANOVA and Variance Results
**Table 2: Comprehensive Statistical Results (coverage and stride ANOVAs, mean drop, Levene, effect sizes)**

| Dataset | Arch | Coverage F (df) | p-value | η² | Stride F (df) | p-value | η² | Mean Δ (100→25) ± σ | Levene p | Cohen's d (aliasing) | Cohen's d (stride) |
|---------|------|-----------------|---------:|----:|---------------|--------:|----:|---------------------:|---------:|---------------------:|--------------------:|
| UCF-101 | TimeSformer | F(4,500)=8.138 | <0.001 | 0.061 | F(4,500)=0.477 | 0.7530 | 0.0038 | 0.0699 ± 0.1112 | 0.0020 | 0.630 | 0.134 |
| UCF-101 | VideoMAE | F(4,500)=32.455 | <0.001 | 0.206 | F(4,500)=13.005 | <0.001 | 0.0942 | 0.1822 ± 0.1861 | <0.001 | 1.380 | 0.763 |
| UCF-101 | ViViT | F(4,500)=20.941 | <0.001 | 0.1435 | F(4,500)=0.792 | 0.5310 | 0.0063 | 0.1302 ± 0.1521 | <0.001 | 1.050 | 0.224 |
| Kinetics-400 | TimeSformer | F(4,1995)=78.770 | <0.001 | 0.1364 | F(4,1995)=0.028 | 0.9985 | 0.0001 | 0.1059 ± 0.0741 | 0.0109 | 1.043 | 0.006 |
| Kinetics-400 | VideoMAE | F(4,1995)=65.984 | <0.001 | 0.1168 | F(4,1995)=0.085 | 0.9871 | 0.0002 | 0.0715 ± 0.0701 | 0.0010 | 0.827 | 0.037 |
| Kinetics-400 | ViViT | F(4,1995)=38.816 | <0.001 | 0.0722 | F(4,1995)=0.089 | 0.9859 | 0.0002 | 0.0824 ± 0.0637 | 0.0294 | 0.782 | 0.036 |

> Note: Mean Δ is the average drop in accuracy from 100% to 25% coverage across classes; Levene p reports the test for variance homogeneity across coverage levels.

**Interpretation**: Coverage has a highly significant main effect on accuracy across all architectures and datasets (all p < 0.001). Effect sizes (η²) vary across dataset–architecture pairs: the largest coverage effects occur for UCF-101 VideoMAE (η² = 0.206) and UCF-101 ViViT (η² = 0.143), while UCF-101 TimeSformer shows a modest coverage effect (η² = 0.061). Stride effects are generally negligible at full coverage (small η² and non-significant p-values) except for **UCF-101 VideoMAE** (F = 13.005, p < 0.001), which shows a meaningful stride dependence.

### 2.4.3 Pairwise Coverage Comparisons (Welch's t-tests)
We computed pairwise Welch's t-tests for all coverage transitions using per-class accuracies (stride = 1). Representative results for **UCF-101 TimeSformer** (n = 101 classes) are:

- 10% vs 25%: t = -2.66, p = 0.00846, d = -0.38 (ns after Bonferroni)
- 10% vs 50%: t = -3.75, p < 0.001, d = -0.53 (medium)
- 10% vs 75%: t = -4.46, p < 0.001, d = -0.63 (medium–large)
- 10% vs 100%: t = -4.46, p < 0.001, d = -0.63 (medium–large)
- 25% vs 100%: t = -1.87, p = 0.06305, d = -0.26 (ns)

For **UCF-101 VideoMAE** pairwise comparisons show even larger effects at low coverage (e.g., 10% vs 100%: t ≈ -9.78, p < 0.001, d ≈ -1.38). Full pairwise tables are available in `evaluations/pairwise_coverage_results.json`.

**Pattern**: The pairwise tests confirm rapid degradation at low coverage and relative stability at high coverage. For **UCF-101 TimeSformer**, Bonferroni-corrected significance (α = 0.005) retains the most severe low-coverage transitions **involving 10% (10%→50%, 10%→75%, 10%→100%)**, whereas moderate transitions such as 10%→25% or 25%→100% do not survive the correction.

### 2.4.4 Variance Heterogeneity
Levene's tests indicate significant heterogeneity of variances across coverage levels for most dataset–architecture combinations (e.g., UCF-101 VideoMAE: Levene p < 1e-20), confirming that variance increases as coverage decreases. This supports our observation that class-level temporal requirements drive heterogeneous aliasing sensitivity (illustrated in Figure 6).

![Figure 6: Variance Analysis](../evaluations/kinetics400/timesformer/per_class_distribution_by_coverage.png)

**Figure 6.** Distribution of per-class accuracies at stride=1 across coverage levels. Left: boxplot showing median, quartiles, and outliers. Right: violin plot revealing increasing spread at reduced coverage.

## 2.5 Action Frequency Taxonomy

Based on empirical aliasing sensitivity, we propose a three-tier motion-frequency taxonomy:

**Table 5: Action Taxonomy by Aliasing Sensitivity**

| Tier | $\Delta$ Threshold | Count | Exemplars | Motion Characteristics |
|------|-------------------|-------|-----------|------------------------|
| High-Sensitivity | $\Delta > 20\%$ | 107 | diving cliff, clean and jerk, vault | High-velocity, explosive motions |
| Moderate-Sensitivity | $10\% < \Delta \leq 20\%$ | 193 | flying kite, breakdancing, snowmobiling | Dynamic controlled motion |
| Low-Sensitivity | $\Delta \leq 10\%$ | 100 | massaging, swinging legs, robot dancing | Gentle, rhythmic, or mechanical motion |

Figure 7 visualizes mean accuracy trajectories for each tier with error bands.

![Figure 7: Sensitivity Tiers](../evaluations/kinetics400/timesformer/per_class_sensitivity_tiers.png)

**Figure 7.** Action classes grouped by aliasing sensitivity tier. High-sensitivity tier (107 classes, $\Delta > 20\%$) exhibits significant degradation below 75% coverage. Moderate-sensitivity tier (193 classes) degrades predictably with coverage reduction. Low-sensitivity tier (100 classes) maintains >70% accuracy even at 10% coverage, demonstrating robustness to aggressive temporal undersampling. Error bands represent ±1 standard deviation within each tier.

---

## 3. Discussion

### 3.1 Executive Summary of Findings 🔍

The empirical results show clear, reproducible patterns: temporal coverage is a primary determinant of recognition accuracy, with the largest declines observed at low coverage levels and significant per-class heterogeneity. VideoMAE exhibits the largest mean aliasing drop on UCF-101, TimeSformer is the most stride-robust at full coverage, and ViViT shows intermediate behavior with occasional paradoxical improvements for structured actions. The following sections interpret these outcomes in terms of architectural design, signal properties of actions, and practical system impacts.

### 3.2 Architecture-level Interpretations 🔧

#### TimeSformer — Attention-driven temporal aggregation
- **Strengths**: Factorized spatiotemporal attention enables flexible, global temporal aggregation. This supports resilience to moderate subsampling because attention can re-weight informative frames and integrate temporal context across longer spans.
- **Weaknesses**: When coverage is severely reduced, the intrinsic temporal context is lost and attention cannot recover missing high-frequency content; thus TimeSformer still experiences notable aliasing on explosive motions.
- **Why results look like this**: The attention mechanism reduces sensitivity to local stride choices (hence low stride F), but it relies on having representative temporal cues distributed through the clip — reducing coverage below the action's critical frequency removes those cues.

#### VideoMAE — Masked autoencoding and temporal reconstruction
- **Strengths**: Masked autoencoder pretraining yields powerful representations that boost peak accuracy when sufficient temporal information is present.
- **Weaknesses**: Strong dependence on dense temporal context makes VideoMAE vulnerable to undersampling and stride changes — the model was trained to reconstruct and predict missing patches in time, so aggressive temporal thinning removes the prediction context and causes large drops in accuracy (observed large η² and Cohen's d for aliasing).
- **Why results look like this**: VideoMAE learns fine-grained temporal correlations; when coverage is low these learned correlations break down leading to larger aliasing sensitivity and stride dependence.

#### ViViT — Local spatiotemporal structure with convolutional inductive bias
- **Strengths**: ViViT's convolutional front-end and factorized attention capture local motion patterns effectively, which helps for structured, rhythmic, or phase-based actions where coarse sampling can still preserve distinguishing patterns.
- **Weaknesses**: Lacks the global attention flexibility of TimeSformer and the reconstruction pressure of VideoMAE, so it sits between the other two in both peak accuracy and aliasing sensitivity.
- **Paradoxical improvements**: For some phase-based actions, moderate undersampling reduces nuisance variation and emphasizes discriminative phase transitions, explaining observed improvements for specific classes.

### 3.3 Signal-level Explanations and Per-class Heterogeneity 📈

- **Motion spectral content**: Actions differ in their dominant temporal frequencies. High-frequency, ballistic, or oscillatory motions contain spectral energy above the effective Nyquist limit of low-coverage configurations and therefore alias when undersampled.
- **Heterogeneous responses**: Levene tests and per-class σ show variance explosion as coverage decreases. This indicates that some classes retain discriminative low-frequency cues, while others require dense sampling; hence global statistics (mean drops) mask large class-level differences.
- **Dataset effects**: UCF-101 (smaller, narrower) shows larger per-class variance for some architectures than Kinetics-400 (broader, more diverse), affecting observed η² and pairwise outcomes.

### 3.4 Practical Implications & System Design Recommendations ⚙️

- **Architecture selection by use-case**:
  - **VideoMAE**: Best for high-accuracy offline/centralized systems where dense, high-fidelity temporal data is available (e.g., clinical video analytics, archival processing). Not recommended for severely bandwidth-constrained or low-frame-rate deployments without retraining/augmentation.
  - **TimeSformer**: Strong choice for applications requiring robustness to variable temporal sampling (e.g., in-the-wild monitoring, driver/passenger monitoring) because attention can integrate sparse but informative frames.
  - **ViViT**: Appropriate for resource-constrained, low-latency scenarios (e.g., edge devices, robotics) where structured/phase-based actions are common and occasional paradoxical improvements may be exploited.

- **Adaptive sampling policies**: Implement action-aware or confidence-driven sampling: use lightweight pre-filters to detect high-frequency activity and switch to high-coverage modes, otherwise operate at reduced coverage for efficiency.

- **Training mitigation strategies**: To reduce aliasing vulnerability, apply temporal augmentation (resampling, frame jitter), multi-scale temporal pretraining, and masked-prediction tasks with variable masking ratios — these reduce reliance on dense temporal correlations.

### 3.5 Limitations and Future Work 🔭

- **Frequency-domain validation**: Systematically measure per-class temporal spectra (optical flow / motion energy) to verify critical-frequency thresholds and guide adaptive sampling policies.
- **Wider architecture coverage**: Extend evaluation to MViT, Video Swin, and hybrid models to generalize architecture–aliasing insights and test whether observed patterns hold.

---

## 4. Conclusion

This comprehensive study establishes temporal sampling as a fundamental consideration in human action recognition system design, with coverage accounting for ~6.1–20.6% of recognition accuracy variance across modern architectures. Our empirical validation of Nyquist-Shannon sampling theory reveals heterogeneous temporal requirements across action categories, with high-frequency motions requiring dense sampling while low-frequency activities remain robust to aggressive temporal reduction.

The identification of architecture-specific temporal processing characteristics provides practical guidelines for optimizing spatiotemporal resolution in resource-constrained applications. TimeSformer demonstrates superior efficiency at high accuracy levels, while VideoMAE and ViViT offer balanced performance for diverse computational constraints.

These findings enable principled system design decisions, moving beyond empirical hyperparameter tuning toward signal processing-guided temporal optimization. The established framework supports the development of adaptive, efficient human action recognition systems capable of real-time operation across healthcare, robotics, autonomous systems, and smart environments.

---

## References

[1] G. Bertasius, H. Wang, and L. Torresani, "Is space-time attention all you need for video understanding?" arXiv preprint arXiv:2102.05095, 2021.

[2] Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked autoencoders for video distribution learning," arXiv preprint arXiv:2203.12602, 2022.

[3] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lucic, and C. Schmid, "ViViT: A video vision transformer," arXiv preprint arXiv:2103.15691, 2021.

[4] C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast networks for video recognition," arXiv preprint arXiv:1812.03982, 2018.

[5] C. Yang, Y. Xu, J. Shi, B. Dai, and B. Zhou, "Temporal pyramid network for action recognition," arXiv preprint arXiv:2004.03548, 2020.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," Advances in neural information processing systems, vol. 30, 2017.

[7] K. Soomro, A. R. Zamir, and M. Shah, "UCF101: A dataset of 101 human actions classes from videos in the wild," arXiv preprint arXiv:1212.0402, 2012.

[8] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, M. Suleyman, and A. Zisserman, "The kinetics human action video dataset," arXiv preprint arXiv:1705.06950, 2017.

---

## Supplementary Material

### Supplementary Figures: Stride-Accuracy Heatmaps

Below are the stride-accuracy heatmaps referenced in Figure 2 (main text). These visualizations show accuracy (color) across coverage × stride grids for each dataset and architecture.

![UCF-101 TimeSformer Heatmap](../evaluations/ucf101/timesformer/accuracy_heatmap.png)

![UCF-101 VideoMAE Heatmap](../evaluations/ucf101/videomae/accuracy_heatmap.png)

![UCF-101 ViViT Heatmap](../evaluations/ucf101/vivit/accuracy_heatmap.png)

![Kinetics-400 TimeSformer Heatmap](../evaluations/kinetics400/timesformer/accuracy_heatmap.png)

![Kinetics-400 VideoMAE Heatmap](../evaluations/kinetics400/videomae/accuracy_heatmap.png)

![Kinetics-400 ViViT Heatmap](../evaluations/kinetics400/vivit/accuracy_heatmap.png)

---
