# Temporal Sampling Results Summary
Source CSV: data/Kinetics400_data/results/timesformer/timesformer-base-finetuned-k400_temporal_sampling.csv

## Best Overall Configuration
- **Accuracy**: 0.7419
- **Coverage**: 100%
- **Stride**: 4
- **Avg Time/Sample**: 0.0000s

## Best Efficiency Configuration (Accuracy per Second)
- **Accuracy/Sec**: 0.000
- **Accuracy**: 0.0000
- **Coverage**: 0%
- **Stride**: 0
- **Avg Time/Sample**: 0.0000s

## Best Configuration per Stride
- **Stride 1**: coverage=100% → accuracy=0.7393
- **Stride 2**: coverage=100% → accuracy=0.7397
- **Stride 4**: coverage=100% → accuracy=0.7419
- **Stride 8**: coverage=100% → accuracy=0.7414
- **Stride 16**: coverage=100% → accuracy=0.7382

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 75% | 16 | 0.7318 | 0.0000 |
| 100% | 16 | 0.7382 | 0.0000 |
| 100% | 8 | 0.7414 | 0.0000 |
| 100% | 4 | 0.7419 | 0.0000 |
