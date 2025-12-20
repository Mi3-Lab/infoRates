import argparse
import os
from info_rates.data.ucf101 import extract_ucf101


def main():
    parser = argparse.ArgumentParser(description="Extract UCF101 dataset and splits.")
    parser.add_argument("--rar", required=True, help="Path to UCF101.rar")
    parser.add_argument("--splits", required=True, help="Path to UCF101TrainTestSplits-RecognitionTask.zip")
    parser.add_argument("--out", default="UCF101_data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    extract_ucf101(args.rar, args.splits, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
