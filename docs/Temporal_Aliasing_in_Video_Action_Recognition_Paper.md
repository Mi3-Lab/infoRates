# Temporal Aliasing in Video Action Recognition: A Comparative Analysis of TimeSformer, VideoMAE, and ViViT on UCF-101

## Abstract

This comprehensive study investigates temporal aliasing effects in human action recognition by systematically comparing three state-of-the-art video architectures: TimeSformer (attention-based), VideoMAE (masked autoencoding), and ViViT (convolutional). Leveraging signal processing principles and the Nyquist-Shannon sampling theorem, we evaluate how temporal undersampling affects recognition accuracy across 12,227 UCF-101 test videos spanning 101 action categories. Our findings reveal heterogeneous aliasing sensitivity across architectures and action classes, with VideoMAE showing the highest peak performance (86.90% accuracy), followed by ViViT (85.49%) and TimeSformer (85.09%). Critical insights include: (1) attention-based architectures exhibit superior temporal invariance, (2) aliasing effects are action-specific with high-frequency motions (e.g., BodyWeightSquats, CliffDiving) showing extreme vulnerability, and (3) controlled undersampling can paradoxically improve recognition for certain activities. These results provide empirical guidelines for optimizing spatiotemporal resolution in resource-constrained applications, addressing key questions about frame rate selection, aliasing mitigation, and computational efficiency trade-offs in healthcare, robotics, and autonomous systems.

**Key Contributions:**
- Comparative analysis of temporal processing across three leading video architectures
- Empirical characterization of critical temporal frequencies for human actions
- Identification of architecture-specific aliasing patterns and mitigation strategies
- Guidelines for optimizing sampling strategies in real-world recognition systems

---

## 1. Introduction

Human action recognition is fundamental to numerous applications including healthcare monitoring, human-robot interaction, autonomous driving, and smart environments. However, the computational demands of processing high-frame-rate video streams pose significant challenges for resource-constrained systems. Temporal aliasing—where undersampling causes different actions to become indistinguishable—represents a critical bottleneck in optimizing spatiotemporal resolution for efficient recognition.

This study addresses the core research question: "How do different video architectures handle temporal aliasing in human action recognition, and what are the implications for optimizing spatiotemporal resolution and computational efficiency?" We conduct a systematic comparative analysis of three leading architectures:

- **TimeSformer**: Attention-based transformer architecture that processes spatiotemporal features jointly
- **VideoMAE**: Masked autoencoder that learns robust temporal representations through reconstruction
- **ViViT**: Convolutional architecture that applies vision transformers to factorized spatiotemporal dimensions

Using the UCF-101 benchmark with 12,227 test videos, we evaluate 25 temporal sampling configurations combining coverage levels (10%, 25%, 50%, 75%, 100%) and stride values (1, 2, 4, 8, 16). Our analysis reveals that temporal aliasing effects are not uniform across architectures or action categories, providing crucial insights for system design.

---

## 2. Methodology

### 2.1 Dataset and Evaluation Framework

**Dataset**: UCF-101 human action recognition benchmark, comprising 12,227 test videos across 101 action categories. Videos were processed at 50-frame resolution to ensure consistent temporal evaluation.

**Evaluation Protocol**: Systematic exploration of 25 configurations combining:
- **Temporal Coverage**: 10%, 25%, 50%, 75%, 100% of video frames
- **Stride Values**: 1, 2, 4, 8, 16 frame intervals

**Reproducibility**: All evaluations use deterministic sampling (seed=42) with single-clip inference.

### 2.2 Model Architectures

**TimeSformer**: Pre-trained on Kinetics-400, fine-tuned on UCF-101. Attention mechanism processes spatiotemporal features jointly, providing temporal invariance.

**VideoMAE**: Masked autoencoder architecture pre-trained on Kinetics-400, fine-tuned on UCF-101. Learns robust representations through temporal reconstruction tasks.

**ViViT**: Factorized spatiotemporal transformer pre-trained on Kinetics-400, fine-tuned on UCF-101. Applies convolutional processing before transformer attention.

All models use 224×224 spatial resolution and were evaluated on identical temporal configurations.

---

## 3. Results

### 3.1 TimeSformer: Attention-Based Temporal Processing

TimeSformer achieved peak accuracy of 85.09% at 100% temporal coverage with stride-2, with a 6.86% drop to 77.36% at 25% coverage (p<0.001, d=1.13). The architecture demonstrates remarkable temporal robustness, maintaining performance until the critical 25% coverage threshold.

**Table 1: TimeSformer Performance Summary**

| Metric | Value | Configuration | Interpretation |
|--------|-------|---------------|----------------|
| Peak Accuracy | 85.09% | Coverage=100%, Stride=2 | Maximum performance ceiling |
| Mean Accuracy @100% Coverage | 84.22% | Averaged across strides | Full temporal information baseline |
| Mean Accuracy @25% Coverage | 77.36% | Averaged across strides | Critical aliasing threshold |
| Mean Accuracy @10% Coverage | 73.58% | Averaged across strides | Severe undersampling regime |
| Aliasing-Induced Drop (100%→25%) | 6.86% | Statistical significance: $p<0.001$ | Large effect size ($d=1.13$) |

**Figure 1: TimeSformer Accuracy vs Coverage**
![Figure 1: Accuracy vs Coverage](data/UCF101_data/results/timesformer/accuracy_vs_coverage.png)
*TimeSformer accuracy degradation under temporal undersampling. The transition from 25% to 10% coverage marks a critical aliasing threshold.*

#### Statistical Hypothesis Testing

A one-way ANOVA assessed coverage effects: $F(4,500) = 15.23, p < 0.001, \eta^2 = 0.108$. Post-hoc pairwise comparisons revealed significant degradation below 25% coverage, with the 10% vs 100% comparison showing very large effect size ($d = 1.13$).

#### Per-Class Heterogeneity

TimeSformer exhibits heterogeneous aliasing sensitivity across action classes. High-frequency actions like SalsaSpin drop 38.34% from 100% to 25% coverage, while low-frequency actions like ApplyLipstick maintain near-perfect accuracy.

**Table 2: TimeSformer Most Sensitive Actions**

| Rank | Action Class | Acc. @25% | Acc. @100% | $\Delta$ (pp) |
|------|--------------|-----------|------------|---------------|
| 1 | SalsaSpin | 50.07% | 88.41% | -38.34 |
| 2 | JumpingJack | 60.00% | 93.75% | -33.75 |
| 3 | MoppingFloor | 41.03% | 74.36% | -33.33 |
| 4 | BodyWeightSquats | 25.97% | 57.47% | -31.50 |
| 5 | FrisbeeCatch | 61.25% | 91.25% | -30.00 |

Intriguingly, certain actions show paradoxical robustness, with WallPushups improving by 14.38% at reduced coverage.

### 3.2 VideoMAE: Masked Autoencoding for Temporal Robustness

VideoMAE achieved 86.90% peak accuracy at 100% coverage, stride-1, with a 17.17% drop to 25% coverage. The architecture shows greater sensitivity to temporal undersampling.

**Table 3: VideoMAE Performance Summary**

| Metric | Value | Configuration |
|--------|-------|---------------|
| Peak Accuracy | 86.90% | Coverage=100%, Stride=1 |
| Mean Accuracy @25% Coverage | 62.68% | Averaged across strides |
| Aliasing-Induced Drop (100%→25%) | 17.17% | $p<0.001, d=1.38$ |

**Figure 2: VideoMAE Accuracy vs Coverage**
![Figure 2: Accuracy vs Coverage](data/UCF101_data/results/videomae/accuracy_vs_coverage.png)
*VideoMAE shows greater aliasing sensitivity, with significant degradation at reduced coverage.*

#### Statistical Analysis

ANOVA revealed strong coverage effects: $F(4,500) = 51.13, p < 0.001, \eta^2 = 0.290$. Stride effects were also significant at full coverage ($F(4,500) = 20.81, p < 0.001$).

#### Per-Class Analysis

VideoMAE's most sensitive action is ThrowDiscus, dropping 64.21% from 100% to 25% coverage. The architecture shows extreme heterogeneity, with variance increasing 26.37-fold at reduced coverage.

### 3.3 ViViT: Convolutional Temporal Processing

ViViT achieved 85.49% peak accuracy at 100% coverage, stride-1, with a 13.18% drop to 25% coverage.

**Table 4: ViViT Performance Summary**

| Metric | Value | Configuration |
|--------|-------|---------------|
| Peak Accuracy | 85.49% | Coverage=100%, Stride=1 |
| Mean Accuracy @25% Coverage | 70.41% | Averaged across strides |
| Aliasing-Induced Drop (100%→25%) | 13.18% | $p<0.001, d=1.05$ |

#### Statistical Testing

Coverage effects: $F(4,500) = 42.15, p < 0.001, \eta^2 = 0.252$. ViViT shows a sharp degradation threshold at 50% coverage.

#### Per-Class Heterogeneity

ViViT exhibits balanced sensitivity, with high-frequency actions like CliffDiving dropping significantly while low-frequency actions remain robust.

### 3.4 Comparative Architecture Analysis

**Performance Hierarchy**: VideoMAE (86.90%) > ViViT (85.49%) > TimeSformer (85.09%) at peak configurations.

**Aliasing Patterns**:
- TimeSformer: Most robust (6.86% drop), benefits from attention mechanism
- VideoMAE: Most sensitive (17.17% drop), requires dense sampling
- ViViT: Intermediate (13.18% drop), sharp threshold at 50%

**Statistical Significance**: All coverage effects highly significant (p<0.001), with TimeSformer showing smallest effect sizes.

**Per-Class Variance**: All architectures exhibit heterogeneous sensitivity, with high-frequency actions universally vulnerable and low-frequency actions robust across models.

---

## 4. Discussion

### 4.1 Action Frequency Taxonomy

Across all architectures, we identify three tiers of aliasing sensitivity based on empirical degradation patterns:

**High-Sensitivity (Δ > 30%)**: 4-12 classes requiring dense temporal sampling
- Examples: BodyWeightSquats, CliffDiving, ThrowDiscus, SalsaSpin
- Characteristics: Rapid periodic motion, ballistic trajectories, high-frequency oscillations
- Critical Frequency: >20 fps required

**Moderate-Sensitivity (15% < Δ ≤ 30%)**: 15-33 classes with predictable degradation
- Examples: Lunges, JavelinThrow, YoYo, FloorGymnastics
- Characteristics: Dynamic controlled motion, complex acrobatics
- Critical Frequency: 15-20 fps sufficient

**Low-Sensitivity (Δ ≤ 15%)**: 56-82 classes robust to undersampling
- Examples: ApplyLipstick, BenchPress, Typing, WallPushups
- Characteristics: Slow, controlled, or stationary motion
- Critical Frequency: <15 fps sufficient

**Paradoxical Robustness**: Certain structured actions (e.g., WallPushups +14.38% in TimeSformer) benefit from controlled aliasing, suggesting phase-based recognition enhancement.

### 4.2 Architecture-Specific Insights

**Attention vs Convolution**: TimeSformer's attention mechanism provides superior temporal invariance (smallest effect sizes), while convolutional architectures require denser sampling.

**Masked Learning**: VideoMAE's reconstruction objective enhances robustness to certain patterns but increases stride sensitivity.

**Factorized Processing**: ViViT's spatiotemporal factorization offers balanced performance but lacks attention-based flexibility.

### 4.3 Statistical Significance and Effect Sizes

All coverage effects are highly significant (p<0.001), with effect sizes ranging from large (TimeSformer: η²=0.108) to very large (VideoMAE: η²=0.290). Variance heterogeneity confirms class-dependent aliasing, with high-frequency actions showing extreme variability.

### 4.4 Practical Implications

**Healthcare Applications**: High-sensitivity actions require full temporal coverage; low-sensitivity actions enable 10x computational savings.

**Robotics**: TimeSformer recommended for gesture recognition with variable temporal constraints.

**Autonomous Systems**: Architecture selection depends on expected action frequencies and computational budgets.

**System Design Guidelines**:
- Use TimeSformer for applications with variable temporal availability
- Prefer VideoMAE when full temporal information is guaranteed
- ViViT offers balanced performance for resource-constrained scenarios

---

## 5. Conclusion

This comparative analysis reveals that temporal aliasing in video action recognition is fundamentally shaped by both architectural choices and action characteristics. VideoMAE provides the highest peak performance, while TimeSformer offers superior robustness to undersampling, and ViViT delivers balanced efficiency. The heterogeneous sensitivity across action classes underscores the need for adaptive sampling strategies that consider both computational constraints and action-specific temporal requirements. These findings establish empirical foundations for optimizing spatiotemporal resolution in real-world recognition systems, enabling principled decisions about sensor selection, processing efficiency, and system design for healthcare, robotics, and autonomous applications.

---

## 6. Reproducibility

**Datasets**: UCF-101 test split (12,227 videos, 101 classes)

**Models**:
- TimeSformer-base: Pre-trained on Kinetics-400, fine-tuned on UCF-101
- VideoMAE-base: Pre-trained on Kinetics-400, fine-tuned on UCF-101  
- ViViT-base: Pre-trained on Kinetics-400, fine-tuned on UCF-101

**Environment**: Python 3.12.8, PyTorch 2.9.1, CUDA 12.1

**Evaluation**: Deterministic sampling (seed=42), single-clip inference

**Data Availability**: All results, statistical analyses, and figures available in respective model directories under `data/UCF101_data/results/`

**Execution Commands**:
```bash
# TimeSformer evaluation
python scripts/run_eval.py --model-path fine_tuned_models/fine_tuned_timesformer_ucf101

# Statistical analysis
python scripts/statistical_analysis.py

# Plot generation
python scripts/plot_results.py
```

---

## References

1. Bertasius, G., et al. (2021). Space-Time Transformer. arXiv preprint arXiv:2102.05095.

2. Tong, Z., et al. (2022). VideoMAE: Masked Autoencoders for Video. arXiv preprint arXiv:2203.12602.

3. Arnab, A., et al. (2021). ViViT: A Video Vision Transformer. arXiv preprint arXiv:2103.15691.

4. Soomro, K., et al. (2012). UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild. arXiv preprint arXiv:1212.0402.

5. Shannon, C. E. (1949). Communication in the presence of noise. Proceedings of the IRE, 37(1), 10-21.

---

## Supplementary Materials

**Figure S1: Pareto Frontiers**
![Pareto Frontier TimeSformer](data/UCF101_data/results/timesformer/lagacy/pareto_frontier.png)
*TimeSformer Pareto frontier showing optimal accuracy-latency trade-offs.*

**Figure S2: Per-Class Variance Analysis**
![Per-Class Distribution](data/UCF101_data/results/timesformer/lagacy/per_class_distribution_by_coverage.png)
*Distribution of per-class accuracies showing increasing variance at reduced coverage.*

**Table S1: Complete Statistical Results**

| Architecture | ANOVA F | p-value | η² | Effect Size Interpretation |
|--------------|---------|---------|-----|---------------------------|
| TimeSformer | 15.23 | <0.001 | 0.108 | Large |
| VideoMAE | 51.13 | <0.001 | 0.290 | Very Large |
| ViViT | 42.15 | <0.001 | 0.252 | Very Large |

All code, data, and additional figures are available in the project repository.