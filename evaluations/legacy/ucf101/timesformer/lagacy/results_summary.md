# Temporal Sampling Results Summary
Source CSV: UCF101_data/results/ucf101_50f_finetuned.csv

## Best Overall Configuration
- **Accuracy**: 0.9843
- **Coverage**: 100%
- **Stride**: 8
- **Avg Time/Sample**: 0.0174s

## Best Efficiency Configuration (Accuracy per Second)
- **Accuracy/Sec**: 57.68
- **Accuracy**: 0.9818
- **Coverage**: 100%
- **Stride**: 1
- **Avg Time/Sample**: 0.0170s

## Best Configuration per Stride
- **Stride 1**: coverage=100% → accuracy=0.9818
- **Stride 2**: coverage=100% → accuracy=0.9829
- **Stride 4**: coverage=100% → accuracy=0.9826
- **Stride 8**: coverage=100% → accuracy=0.9843
- **Stride 16**: coverage=100% → accuracy=0.9800

## Pareto-Optimal Frontier (Accuracy vs Latency)
Configurations where accuracy cannot be improved without increasing latency.

| Coverage | Stride | Accuracy | Avg Time (s) |
|----------|--------|----------|______________|
| 10% | 8 | 0.7950 | 0.0170 |
| 100% | 1 | 0.9818 | 0.0170 |
| 100% | 4 | 0.9826 | 0.0170 |
| 100% | 2 | 0.9829 | 0.0171 |
| 100% | 8 | 0.9843 | 0.0174 |
