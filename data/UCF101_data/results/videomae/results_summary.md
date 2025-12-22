# Temporal Sampling Results Summary
Source CSV: data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_temporal_sampling.csv

## Best Overall Configuration
- **Accuracy**: 0.9464
- **Coverage**: 100%
- **Stride**: 1
- **Avg Time/Sample**: 0.0288s

## Best Efficiency Configuration (Accuracy per Second)
- **Accuracy/Sec**: 33.28
- **Accuracy**: 0.9317
- **Coverage**: 75%
- **Stride**: 1
- **Avg Time/Sample**: 0.0280s

## Best Configuration per Stride
- **Stride 1**: coverage=100% → accuracy=0.9464
- **Stride 2**: coverage=100% → accuracy=0.9443
- **Stride 4**: coverage=50% → accuracy=0.8863
- **Stride 8**: coverage=100% → accuracy=0.9007
- **Stride 16**: coverage=100% → accuracy=0.8381

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 75% | 1 | 0.9317 | 0.0280 |
| 100% | 1 | 0.9464 | 0.0288 |
