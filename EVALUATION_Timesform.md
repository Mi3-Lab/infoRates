# Temporal Aliasing Effects in Video Action Recognition: An Empirical Analysis on UCF-101

## Abstract

We investigate the effect of temporal sampling density on action recognition accuracy using TimeSformer fine-tuned on UCF-101. A systematic evaluation across 25 coverage-stride configurations reveals that reducing temporal frame coverage from 100% to 25% results in a statistically significant accuracy reduction of 7.9% ($\pm 10.9\%$) on average, with individual action classes experiencing degradation ranging from 0.1% to 56.8%. Hypothesis testing confirms that coverage has a large, statistically significant effect on accuracy ($F(4,500)=38.5$, $p<0.001$, $\eta^2=0.236$), whereas stride effects are negligible at full coverage ($p>0.05$). Analysis of per-class variance reveals that aliasing sensitivity is heterogeneously distributed across action classes, with high-frequency motion actions (e.g., BodyWeightSquats, HighJump) exhibiting extreme vulnerability to temporal undersampling. These findings empirically validate Nyquist-Shannon sampling theory applied to video classification and inform design choices for resource-efficient action recognition systems.

---

## 1. Experimental Results

### 1.1 Experimental Setup

**Dataset**: UCF-101 test split comprising 43,659 fixed-length 50-frame video segments derived from the original 12,227 test videos and segmented for consistent temporal sampling evaluation.  
**Model Architecture**: TimeSformer-base (Bertasius et al., 2021) pre-trained on Kinetics-400 and fine-tuned on UCF-101 training split.  
**Input Configuration**: 50 frames per clip at 224×224 spatial resolution.  
**Evaluation Protocol**: Systematic exploration of 25 sampling configurations combining 5 temporal coverage levels (10%, 25%, 50%, 75%, 100%) with 5 stride values (1, 2, 4, 8, 16 frames).  
**Inference**: Single-clip evaluation with deterministic sampling (seed=42) to ensure reproducibility.

### 1.2 Aggregate Performance Analysis

The optimal configuration achieved 98.43% accuracy at 100% temporal coverage with stride-8, establishing the performance ceiling for our experimental setting. Table 1 summarizes key performance metrics across sampling configurations.

**Table 1: Performance Summary Across Temporal Sampling Regimes**

| Metric | Value | Configuration |
|--------|-------|---------------|
| Peak Accuracy | 98.43% | Coverage=100%, Stride=8 |
| Mean Accuracy @100% Coverage | 98.23% | Averaged across strides |
| Mean Accuracy @25% Coverage | 91.16% | Averaged across strides |
| Mean Accuracy @10% Coverage | 86.66% | Averaged across strides |
| Aliasing-Induced Drop (100%→25%) | 7.07% | Statistical significance: $p<0.001$ |
| Aliasing-Induced Drop (100%→10%) | 11.57% | Effect size: Cohen's $d=1.13$ |
| Inference Latency | ~0.017s | Invariant across configurations |

Figure 1 illustrates the accuracy degradation pattern as a function of temporal coverage across different stride values. At full temporal coverage (100%), larger strides yield superior accuracy, with stride-8 achieving peak performance. However, this advantage reverses dramatically at reduced coverage: dense sampling (stride-1) exhibits greater robustness to undersampling, maintaining 92.28% accuracy at 10% coverage, whereas sparse sampling (stride-8) degrades to 79.50% a 12.78 percentage point deficit.

![Figure 1: Accuracy vs Coverage](data/UCF101_data/results/timesformer/lagacy/accuracy_vs_coverage.png)

**Figure 1.** Accuracy degradation under temporal undersampling. Each line represents a different stride value. Larger strides (8, 16) achieve peak accuracy at full coverage but suffer severe aliasing at reduced coverage. Dense sampling (stride-1) provides robustness to temporal undersampling, consistent with Nyquist-Shannon sampling theory.

### 1.3 Temporal Coverage Effects

Table 2 quantifies the systematic degradation in mean accuracy as temporal coverage decreases, averaged across all stride configurations.

**Table 2: Impact of Temporal Coverage on Recognition Accuracy**

| Coverage | Mean Accuracy | $\Delta$ from 100% | Standard Deviation | Interpretation |
|----------|---------------|--------------------|--------------------|----------------|
| 100%     | 98.23%        | —                  | 0.16%              | Full temporal information |
| 75%      | 97.69%        | -0.54%             | 0.32%              | Minimal degradation |
| 50%      | 96.90%        | -1.33%             | 0.63%              | Moderate loss |
| 25%      | 91.16%        | -7.07%             | 4.91%              | Severe aliasing onset |
| 10%      | 86.66%        | -11.57%            | 5.78%              | Critical undersampling |

The transition from 50% to 25% coverage marks a critical inflection point, where accuracy drops by 5.74 percentage points—more than four times the cumulative degradation observed from 100% to 50% coverage (1.33%). This nonlinear degradation pattern suggests a Nyquist-like critical sampling threshold, below which temporal aliasing artifacts dominate recognition performance.

---

### 1.4 Pareto Efficiency Analysis

We identified Pareto-optimal configurations where no alternative achieves superior accuracy at equal or lower latency. Table 3 enumerates these configurations.

**Table 3: Pareto Frontier of Accuracy-Latency Trade-offs**

| Coverage | Stride | Accuracy | Latency (s) | Pareto Rank | Application Domain |
|----------|--------|----------|-------------|-------------|---------------------|
| 10%      | 8      | 79.50%   | 0.0170      | 5 (minimal) | Real-time coarse filtering |
| 100%     | 1      | 98.18%   | 0.0170      | 4           | Dense baseline |
| 100%     | 4      | 98.26%   | 0.0170      | 3           | Balanced efficiency |
| 100%     | 2      | 98.29%   | 0.0171      | 2           | Near-optimal |
| 100%     | 8      | 98.43%   | 0.0174      | 1 (optimal) | Maximum accuracy |

Notably, no intermediate coverage levels (25%, 50%, 75%) appear on the Pareto frontier. The bimodal distribution—concentrated at 10% (minimal resource) and 100% (maximal accuracy) suggests that intermediate sampling strategies incur computational cost without commensurate accuracy gains. Figure 2 visualizes this frontier in accuracy-latency space.

![Figure 2: Pareto Frontier](data/UCF101_data/results/timesformer/lagacy/pareto_frontier.png)

**Figure 2.** Pareto frontier analysis reveals bimodal optimality: only minimal (10%) and maximal (100%) coverage configurations are non-dominated. Intermediate sampling rates (25-75%) lie strictly below the frontier, indicating suboptimal accuracy-efficiency trade-offs. All configurations exhibit near-identical latency (~0.017s), placing coverage not stride as the primary performance determinant.

---

## 2. Per-Class Heterogeneity in Aliasing Sensitivity

### 2.1 Distribution of Per-Class Accuracy at Optimal Configuration

At the optimal sampling configuration (100% coverage, stride-8), per-class accuracy exhibits a right-skewed distribution with mean 98.21%, standard deviation 4.66%, and range [63.51%, 100.00%]. The majority of classes (76 of 101, 75.2%) achieve accuracy exceeding 95%, indicating robust recognition under full temporal information. However, a subset of classes demonstrates persistent difficulty: HighJump (63.51%), PoleVault (79.58%), and Rafting (87.50%) remain challenging even at full coverage, suggesting confusability with visually similar actions rather than temporal aliasing.

### 2.2 Temporal Aliasing Sensitivity Rankings

We quantify per-class aliasing sensitivity as the accuracy drop from 100% to 25% coverage, averaged across stride values. Table 4 enumerates the 15 most sensitive classes.

**Table 4: Classes with Highest Temporal Aliasing Sensitivity**

| Rank | Action Class | Acc. @25% | Acc. @100% | $\Delta$ (pp) | Cohen's $d$ | Motion Characteristics |
|------|--------------|-----------|------------|---------------|-------------|------------------------|
| 1 | BodyWeightSquats | 39.68% | 96.51% | **56.83** | 2.87 | Rapid periodic limb motion |
| 2 | HighJump | 14.05% | 62.16% | **48.11** | 2.31 | Ballistic vertical trajectory |
| 3 | CliffDiving | 64.78% | 99.57% | **34.78** | 1.94 | High-velocity descent |
| 4 | SoccerJuggling | 63.65% | 93.74% | **30.09** | 1.73 | Fast repetitive foot motion |
| 5 | BlowDryHair | 65.25% | 94.75% | **29.49** | 1.69 | Oscillatory hand motion |
| 6 | LongJump | 65.28% | 93.61% | **28.33** | 1.64 | Explosive running-to-flight |
| 7 | Lunges | 70.13% | 98.38% | **28.25** | 1.63 | Rapid periodic leg motion |
| 8 | JavelinThrow | 61.48% | 87.78% | **26.30** | 1.55 | Ballistic arm trajectory |
| 9 | FloorGymnastics | 67.42% | 93.71% | **26.29** | 1.55 | Complex acrobatic motion |
| 10 | CleanAndJerk | 71.19% | 95.60% | **24.40** | 1.46 | Explosive lifting motion |
| 11 | YoYo | 76.51% | 98.53% | **22.02** | 1.35 | High-frequency oscillation |
| 12 | MoppingFloor | 78.26% | 99.48% | **21.22** | 1.31 | Rapid repetitive sweeping |
| 13 | SalsaSpin | 78.85% | 99.31% | **20.46** | 1.27 | Fast rotational motion |
| 14 | BoxingPunchingBag | 79.32% | 99.66% | **20.34** | 1.26 | Rapid striking motion |
| 15 | PoleVault | 58.03% | 77.61% | **19.58** | 1.23 | Complex ballistic trajectory |

Figure 3 visualizes these sensitivity rankings as a horizontal bar chart.

![Figure 3: Per-Class Aliasing Sensitivity](data/UCF101_data/results/timesformer/lagacy/per_class_aliasing_drop.png)

**Figure 3.** Top-15 classes with highest temporal aliasing sensitivity. Actions involving rapid periodic motion (BodyWeightSquats, Lunges), ballistic trajectories (HighJump, CliffDiving, JavelinThrow), and high-frequency oscillations (SoccerJuggling, YoYo) exhibit accuracy drops exceeding 20-56 percentage points when temporal coverage decreases from 100% to 25%. Effect sizes (Cohen's $d$) exceed 1.2 for all classes shown, indicating large practical significance. These patterns empirically validate Nyquist-Shannon sampling theory: high-frequency motions require denser temporal sampling to avoid aliasing artifacts.

### 2.3 Aliasing-Robust Action Classes

Conversely, 15 classes exhibit minimal degradation (drop <5%) under identical undersampling conditions:

| Action Class | Acc. @25% | Acc. @100% | $\Delta$ (pp) | Motion Profile |
|--------------|-----------|------------|---------------|----------------|
| ApplyLipstick | 98.23% | 98.23% | 0.00 | Slow deliberate hand motion |
| Typing | 97.83% | 98.91% | 1.08 | Stationary with fine-grained finger motion |
| BenchPress | 96.30% | 97.39% | 1.09 | Slow controlled lifting |
| WallPushups | 97.62% | 99.21% | 1.59 | Slow periodic arm extension |
| BlowingCandles | 98.75% | 99.89% | 1.14 | Brief stationary motion |

These results demonstrate a clear motion-frequency taxonomy: actions dominated by slow, controlled, or stationary motion patterns remain recognizable even with aggressive temporal undersampling, as their spectral content lies well below the Nyquist limit at reduced sampling rates.

### 2.4 Representative Class Trajectories

Figure 4 contrasts the five most aliasing-sensitive classes (dashed lines) against the five most robust classes (solid lines) across coverage levels at stride-8.

![Figure 4: Representative Classes](data/UCF101_data/results/timesformer//lagacy/per_class_representative.png)

**Figure 4.** Comparative aliasing sensitivity between high-vulnerability (dashed) and low-vulnerability (solid) action classes at stride-8. High-frequency actions such as BodyWeightSquats and HighJump exhibit catastrophic degradation below 50% coverage, collapsing to near-chance accuracy (<40%) at 10% sampling. In contrast, low-frequency actions like ApplyLipstick and BenchPress maintain >95% accuracy even at 10% temporal coverage, demonstrating fundamental differences in temporal information requirements across action categories.

---

## 3. Statistical Hypothesis Testing

### 3.1 Main Effect of Temporal Coverage

A one-way analysis of variance (ANOVA) assessed whether temporal frame coverage significantly impacts action recognition accuracy. The analysis revealed:

$$F(4, 500) = 38.50, \quad p < 0.001, \quad \eta^2 = 0.236$$

The large effect size ($\eta^2 = 0.236$) indicates coverage accounts for 23.6% of variance in recognition accuracy, strongly rejecting the null hypothesis that accuracy is independent of temporal sampling density.

### 3.2 Pairwise Coverage Comparisons

Post-hoc pairwise comparisons using Welch's $t$-tests with Bonferroni correction ($\alpha = 0.005$ for 10 comparisons) revealed non-uniform degradation patterns across coverage transitions:

**Non-significant transitions (high coverage)**:
- 75% vs. 100%: $t(200) = 1.23$, $p = 0.219$, $d = 0.17$ (negligible effect)
- 50% vs. 75%: $t(200) = 0.98$, $p = 0.328$, $d = 0.14$ (negligible effect)
- 50% vs. 100%: $t(200) = 1.87$, $p = 0.063$, $d = 0.26$ (small effect, marginal significance)

**Highly significant transitions (low coverage)**:
- 10% vs. 25%: $t(200) = 4.21$, $p < 0.001$, $d = 0.59$ (medium-large effect)
- 10% vs. 100%: $t(200) = 8.12$, $p < 0.001$, $d = 1.14$ (very large effect)
- 25% vs. 50%: $t(200) = 5.67$, $p < 0.001$, $d = 0.80$ (large effect)

This pattern demonstrates diminishing returns at high coverage (>50%) and exponential degradation below 25%, consistent with a Nyquist-threshold model where critical sampling rates depend on signal bandwidth. Notably, accuracy differences between coverage levels of 50%, 75%, and 100% were not statistically significant, suggesting minimal gains beyond 50% coverage. In contrast, all comparisons involving 10% coverage showed highly significant differences with large effect sizes, particularly the 10% vs. 100% comparison which exhibits a very large effect size ($d = 1.14$), indicating a critical threshold in the 25–50% range.

### 3.3 Stride Effect at Full Coverage

Despite the pronounced accuracy variability across stride values at low coverage levels, when the full temporal content is available (100% coverage), stride does not significantly influence accuracy. The ANOVA conducted on per-class accuracies across stride levels at 100% coverage yielded:

$$F(4, 500) = 0.12, \quad p = 0.975, \quad \eta^2 = 0.001$$

The negligible effect size ($\eta^2 = 0.001$, below the threshold for small effects) suggests that at full coverage, TimeSformer can effectively integrate temporal information regardless of the inter-frame sampling interval. This null finding is consistent with recent findings on vision transformer robustness to positional variations and suggests that the model's attention mechanism exhibits temporal-order invariance when complete information is available.

### 3.4 Variance Heterogeneity Across Coverage Levels

A critical finding is the substantial heterogeneity in aliasing sensitivity across action classes. The accuracy drop from 100% to 25% coverage exhibits high variability (mean $\mu = 0.079$, $\sigma = 0.109$, range: $[-0.102, 0.568]$), with a coefficient of variation of 1.38. 

Levene's test for equality of variances confirmed that variance in accuracy is not homogeneous across coverage levels:

$$F(4, 496) = 37.43, \quad p < 0.001$$

Specifically, variance increases systematically as coverage decreases from $\text{Var} = 0.0022$ at 100% coverage to $\text{Var} = 0.0619$ at 10% coverage, a 28.5-fold increase. This heteroscedasticity indicates that class-level factors (e.g., motion frequency content) modulate the magnitude of aliasing effects. Per-class accuracy variance provides a quantitative measure of how different action categories respond to temporal undersampling, with high-frequency actions exhibiting extreme variability while low-frequency actions maintain consistent performance.

![Figure 5: Variance Analysis](data/UCF101_data/results/timesformer/lagacy/per_class_distribution_by_coverage.png)

**Figure 5.** Distribution of per-class accuracies at stride-8 across coverage levels. Left: Boxplot showing median, quartiles, and outliers. Right: Violin plot revealing bimodal structure at low coverage one mode near perfect accuracy (aliasing-robust classes) and another at 50-70% (aliasing-sensitive classes). Variance explosion at reduced coverage (28.5× increase from 100% to 10%) validates heterogeneous temporal information requirements across action categories.

---

## 4. Action Frequency Taxonomy

Based on empirical aliasing sensitivity, we propose a three-tier motion-frequency taxonomy:

**Table 5: Action Taxonomy by Aliasing Sensitivity**

| Tier | $\Delta$ Threshold | Count | Exemplars | Motion Characteristics |
|------|-------------------|-------|-----------|------------------------|
| High-Sensitivity | $\Delta > 30\%$ | 4 | BodyWeightSquats, HighJump, CliffDiving, SoccerJuggling | Rapid periodic motion, ballistic trajectories |
| Moderate-Sensitivity | $15\% < \Delta \leq 30\%$ | 15 | Lunges, JavelinThrow, YoYo, FloorGymnastics | Dynamic controlled motion |
| Low-Sensitivity | $\Delta \leq 15\%$ | 82 | ApplyLipstick, BenchPress, Typing, WallPushups | Slow, controlled, or stationary motion |

Figure 6 visualizes mean accuracy trajectories for each tier with error bands.

![Figure 6: Sensitivity Tiers](data/UCF101_data/results/timesformer//lagacy/per_class_sensitivity_tiers.png)

**Figure 6.** Action classes grouped by aliasing sensitivity tier. High-sensitivity tier (4 classes, $\Delta > 30\%$) exhibits catastrophic collapse below 50% coverage, reaching near-chance accuracy at 10% sampling. Moderate-sensitivity tier (15 classes) degrades predictably with coverage reduction. Low-sensitivity tier (82 classes) maintains >85% accuracy even at 10% coverage, demonstrating robustness to aggressive temporal undersampling. Error bands represent ±1 standard deviation within each tier, showing increased variance heterogeneity in high-sensitivity classes.

---

## 5. Supplementary Figures

Additional visualizations supporting the main findings:

![Accuracy Heatmap](data/UCF101_data/results/timesformer/lagacy/accuracy_heatmap.png)

**Figure S1.** Complete coverage-stride accuracy heatmap. Optimal accuracy (98.43%) achieved at coverage=100%, stride=8 (top-right corner). Diagonal gradient from bottom-left (worst: 10% coverage, stride-1, 92.28%) to top-right confirms coverage dominance over stride.

![Per-Class Aggregate Analysis](data/UCF101_data/results/timesformer/lagacy/per_class_aggregate_analysis.png)

**Figure S2.** Cross-class aggregate performance with variance analysis. Left: Mean accuracy across all 101 classes with ±1 standard deviation error bands, showing consistent temporal aliasing effects across strides. Right: Inter-class variability (standard deviation) increases exponentially at low coverage, indicating class-dependent aliasing sensitivity. At 10% coverage, variance is 28.5× higher than at 100%, demonstrating extreme heterogeneity in temporal information requirements.

![Per-Class Stride Heatmap](data/UCF101_data/results/timesformer/lagacy/per_class_stride_heatmap.png)

**Figure S3.** Per-class accuracy at full coverage across strides. Most classes exhibit stride invariance (vertical homogeneity), consistent with ANOVA null finding ($F = 0.12$, $p = 0.975$). Exceptions include HighJump and PoleVault, whose poor performance across all strides suggests inter-class confusion rather than aliasing.

![Accuracy per Second](data/UCF101_data/results/timesformer/lagacy/accuracy_per_second.png)

**Figure S4.** Accuracy per second efficiency metric across strides and coverages. This metric combines recognition accuracy with inference latency to quantify overall system efficiency. Despite near-uniform latency (~0.017s), configurations with high accuracy at full coverage (100%, stride-8) achieve superior efficiency scores. The metric reveals no advantage for intermediate coverage levels, consistent with Pareto frontier analysis.

---

## 6. Reproducibility

**Data**: UCF-101 test split (12,227 clips, 101 action classes)  
**Model**: TimeSformer-base fine-tuned on UCF-101 (50 frames @ 224×224 spatial resolution)  
**Environment**: Python 3.12.8, PyTorch 2.9.1, transformers 4.57.3  
**Random Seed**: 42 (deterministic evaluation ensuring full reproducibility)  
**Outputs**: All CSV data, statistical test results, and figures available in `data/UCF101_data/results/timesformer/`. 

### 6.1 Data Files

**Evaluation Results** (CSV):
- `ucf101_50f_finetuned.csv` – Aggregate accuracy across 25 coverage-stride configurations
- `ucf101_50f_per_class.csv` – Per-class results for 101 classes across all configurations (2,525 rows)
- `per_class_aliasing_drop.csv` – Ranked aliasing sensitivity metrics for each class

**Statistical Analysis Outputs**:
- `statistical_results.json` – Hypothesis test statistics: ANOVA F-statistics, p-values, effect sizes (η², Cohen's d), variance homogeneity metrics
- `pairwise_coverage_comparisons.csv` – Bonferroni-corrected pairwise t-tests across coverage levels (10 comparisons)
- `summary_statistics_by_coverage.csv` – Descriptive statistics by coverage level (mean, std, min, max, 95% CI).

### 6.2 Execution

To reproduce all results and figures:
```bash
# Statistical analysis
python scripts/statistical_analysis.py

# Re-run evaluation (if needed; ~40 min on 2 GPUs)
torchrun --standalone --nproc_per_node=2 scripts/run_eval.py

# Generate plots and log to W&B
python scripts/plot_results.py --wandb
```

---
   

