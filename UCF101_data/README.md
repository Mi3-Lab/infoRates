# UCF101 Dataset

This folder contains the UCF101 action recognition dataset, organized and ready for experiments.

## Structure

```
UCF101_data/
├── UCF-101/              # Video files organized by action class
│   ├── ApplyEyeMakeup/
│   ├── ApplyLipstick/
│   ├── Archery/
│   └── ... (101 classes total)
└── ucfTrainTestlist/     # Official train/test split files
    ├── classInd.txt
    ├── trainlist01.txt
    ├── trainlist02.txt
    ├── trainlist03.txt
    ├── testlist01.txt
    ├── testlist02.txt
    └── testlist03.txt
```

## Dataset Details

- **Total classes**: 101 action categories
- **Total videos**: 13,320 videos
- **Video format**: `.avi` files
- **Average duration**: ~7 seconds per video
- **Splits**: 3 official train/test splits provided

## Usage

The scripts in the parent directory automatically use this data:

- `scripts/train_timesformer.py` - Fine-tune models on UCF101
- `scripts/split_fixed.py` - Create fixed-frame clips for evaluation
- `scripts/run_eval.py` - Run temporal sampling experiments

## Adding Other Datasets

To add additional datasets for comparison (e.g., Kinetics, HMDB51, Something-Something-v2):

1. Create a new subdirectory: `mkdir UCF101_data/<dataset_name>`
2. Extract videos into class-based folders
3. Adapt the data loading scripts in `src/info_rates/data/` to support the new dataset

## Notes

- The UCF-101 folder contains the raw videos extracted from `UCF101-001.rar`
- The train/test splits follow the official UCF101 protocol
- For temporal sampling experiments, use `scripts/split_fixed.py` to create uniform-length clips
