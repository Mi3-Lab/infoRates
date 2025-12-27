# Temporal Sampling Results Summary
Source CSV: data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_temporal_sampling.csv

## Best Overall Configuration
- **Accuracy**: 0.8690
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
- **Stride 1**: coverage=100% → accuracy=0.8690
- **Stride 2**: coverage=100% → accuracy=0.8669
- **Stride 4**: coverage=50% → accuracy=0.7845
- **Stride 8**: coverage=100% → accuracy=0.7838
- **Stride 16**: coverage=100% → accuracy=0.7189

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 25% | 16 | 0.4192 | 0.0000 |
| 100% | 16 | 0.7189 | 0.0000 |
| 100% | 8 | 0.7838 | 0.0000 |
| 50% | 4 | 0.7845 | 0.0000 |
| 75% | 2 | 0.8459 | 0.0000 |
| 100% | 2 | 0.8669 | 0.0000 |
| 100% | 1 | 0.8690 | 0.0000 |
