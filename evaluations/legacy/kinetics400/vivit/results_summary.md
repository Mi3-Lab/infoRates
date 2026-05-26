# Temporal Sampling Results Summary
Source CSV: data/Kinetics400_data/results/vivit/vivit-b-16x2_temporal_sampling.csv

## Best Overall Configuration
- **Accuracy**: 0.7619
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
- **Stride 1**: coverage=100% → accuracy=0.7619
- **Stride 2**: coverage=75% → accuracy=0.7577
- **Stride 4**: coverage=100% → accuracy=0.7604
- **Stride 8**: coverage=100% → accuracy=0.7588
- **Stride 16**: coverage=100% → accuracy=0.7558

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 25% | 16 | 0.6781 | 0.0000 |
| 100% | 16 | 0.7558 | 0.0000 |
| 100% | 8 | 0.7588 | 0.0000 |
| 100% | 4 | 0.7604 | 0.0000 |
| 75% | 2 | 0.7577 | 0.0000 |
| 100% | 1 | 0.7619 | 0.0000 |