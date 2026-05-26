# Temporal Sampling Results Summary
Source CSV: data/Kinetics400_data/results/videomae/videomae-base-finetuned-kinetics_temporal_sampling.csv

## Best Overall Configuration
- **Accuracy**: 0.7652
- **Coverage**: 50%
- **Stride**: 2
- **Avg Time/Sample**: 0.0000s

## Best Efficiency Configuration (Accuracy per Second)
- **Accuracy/Sec**: 0.000
- **Accuracy**: 0.0000
- **Coverage**: 0%
- **Stride**: 0
- **Avg Time/Sample**: 0.0000s

## Best Configuration per Stride
- **Stride 1**: coverage=50% → accuracy=0.7646
- **Stride 2**: coverage=50% → accuracy=0.7652
- **Stride 4**: coverage=50% → accuracy=0.7638
- **Stride 8**: coverage=50% → accuracy=0.7598
- **Stride 16**: coverage=50% → accuracy=0.7539

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 50% | 16 | 0.7539 | 0.0000 |
| 50% | 8 | 0.7598 | 0.0000 |
| 50% | 4 | 0.7638 | 0.0000 |
| 50% | 2 | 0.7652 | 0.0000 |
| 50% | 1 | 0.7646 | 0.0000 |