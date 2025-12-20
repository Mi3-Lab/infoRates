# Temporal Sampling Results Summary
Source CSV: UCF101_data/results/ucf101_50f_finetuned.csv
- Best overall: accuracy=0.9843, coverage=100%, stride=8, avg_time=0.0163s
- Best efficiency (accuracy/sec): acc/sec=60.89, accuracy=0.9792, coverage=75%, stride=1, avg_time=0.0161s

- Best per stride:
  - stride=1: coverage=100% → accuracy=0.9818
  - stride=2: coverage=100% → accuracy=0.9829
  - stride=4: coverage=100% → accuracy=0.9826
  - stride=8: coverage=100% → accuracy=0.9843
  - stride=16: coverage=100% → accuracy=0.9800
