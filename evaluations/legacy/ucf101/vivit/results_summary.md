# Temporal Sampling Results Summary
Source CSV: data/UCF101_data/results/vivit/fine_tuned_vivit_ucf101_temporal_sampling.csv

## Best Overall Configuration
- **Accuracy**: 0.8549
- **Coverage**: 100%
- **Stride**: 1
- **Avg Time/Sample**: 0.0000s

## Best Efficiency Configuration (Accuracy per Second)
- **Accuracy/Sec**: 0.000
- **Accuracy**: 0.0000
- **Coverage**: 0%
- **Stride**: 0
- **Avg Time/Sample**: 0.0000s

## Best Configuration per Stride
- **Stride 1**: coverage=100% → accuracy=0.8549
- **Stride 2**: coverage=75% → accuracy=0.8389
- **Stride 4**: coverage=100% → accuracy=0.8440
- **Stride 8**: coverage=100% → accuracy=0.8397
- **Stride 16**: coverage=100% → accuracy=0.8017

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 25% | 16 | 0.5629 | 0.0000 |
| 100% | 16 | 0.8017 | 0.0000 |
| 100% | 8 | 0.8397 | 0.0000 |
| 100% | 4 | 0.8440 | 0.0000 |
| 75% | 1 | 0.8483 | 0.0000 |
| 100% | 1 | 0.8549 | 0.0000 |
