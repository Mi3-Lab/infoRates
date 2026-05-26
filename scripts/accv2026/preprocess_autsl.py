#!/usr/bin/env python3
"""Preprocess AUTSL (Turkish Sign Language) dataset after Kaggle download.

Input:  data/AUTSL_data/
  train/signer*_sample*_color.mp4   (28 142 clips)
  val/signer*_sample*_color.mp4     (4 418 clips)
  train_labels.csv  — no header: sample_name, class_id   (0-225)
  val_labels.csv    — no header: sample_name, class_id

Output:
  data/AUTSL_data/splits/train.csv   (video_path, label, label_id)
  data/AUTSL_data/splits/val.csv
  data/AUTSL_data/splits/classes.csv
"""
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/AUTSL_data")
    args = parser.parse_args()

    root = Path(args.data_root)
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess-autsl] Root: {root}")

    # Load labels — no header, cols: sample_name, class_id
    train_df = pd.read_csv(root / "train_labels.csv", header=None, names=["sample", "label_id"])
    val_df   = pd.read_csv(root / "val_labels.csv",   header=None, names=["sample", "label_id"])

    print(f"  train labels: {len(train_df)}")
    print(f"  val   labels: {len(val_df)}")

    # Class names: use numeric ids (0-225); no gloss file in this Kaggle release
    all_ids = sorted(set(train_df["label_id"].tolist() + val_df["label_id"].tolist()))
    id2name = {i: f"sign_{i:03d}" for i in all_ids}
    print(f"  classes: {len(id2name)}")

    output_rows = {}
    for split_name, df in [("train", train_df), ("val", val_df)]:
        video_dir = root / split_name
        rows, missing = [], 0
        for _, row in df.iterrows():
            sample   = str(row["sample"])
            label_id = int(row["label_id"])
            vid = video_dir / f"{sample}_color.mp4"
            if not vid.exists():
                missing += 1
                continue
            rows.append({
                "video_path": str(vid),
                "label":      id2name[label_id],
                "label_id":   label_id,
            })
        output_rows[split_name] = rows
        print(f"  {split_name}: {len(rows)} found, {missing} missing")

    for split_name, rows in output_rows.items():
        out = splits_dir / f"{split_name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Saved: {out}")

    pd.DataFrame([{"label_id": k, "label": v} for k, v in sorted(id2name.items())]).to_csv(
        splits_dir / "classes.csv", index=False
    )
    print(f"[preprocess-autsl] Done. {len(id2name)} classes.")


if __name__ == "__main__":
    main()
