#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def load_split_basenames(split_file: Path):
    """Load basenames (without extension) from UCF101 split list, e.g., v_ApplyEyeMakeup_g01_c01"""
    allowed = set()
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: ClassName/v_ApplyEyeMakeup_g01_c01.avi
            parts = line.split('/')
            name = parts[-1]
            base = name.rsplit('.', 1)[0]  # remove .avi
            allowed.add(base)
    return allowed


def clip_base_from_filename(fp: Path):
    """Given path like v_ApplyEyeMakeup_g01_c01_seg00.mp4, return v_ApplyEyeMakeup_g01_c01"""
    stem = fp.stem  # v_ApplyEyeMakeup_g01_c01_seg00
    if '_seg' in stem:
        stem = stem.split('_seg', 1)[0]
    return stem


def build_manifest_from_clips(clips_dir: Path, out_csv: Path, split_file: Path | None = None):
    clips_dir = Path(clips_dir)
    rows = []
    allowed = None
    if split_file and split_file.exists():
        allowed = load_split_basenames(split_file)

    for cls_dir in clips_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for fp in cls_dir.glob('*.mp4'):
            if allowed is not None:
                base = clip_base_from_filename(fp)
                if base not in allowed:
                    continue
            rows.append({"video_path": str(fp), "label": label})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote manifest: {out_csv} with {len(df)} rows and {df['label'].nunique()} classes")


def main():
    ap = argparse.ArgumentParser(description="Build manifest CSV from existing fixed clips, optionally filtered by split file")
    ap.add_argument('--clips-dir', required=True, help='Path to fixed clips directory (e.g., UCF101_data/UCF101_50f)')
    ap.add_argument('--out', required=True, help='Output manifest CSV path (e.g., UCF101_data/manifests/ucf101_50f.csv)')
    ap.add_argument('--split-file', default=None, help='Optional UCF101 split file (e.g., ucfTrainTestlist/testlist01.txt) to filter dev/test videos')
    args = ap.parse_args()

    build_manifest_from_clips(Path(args.clips_dir), Path(args.out), Path(args.split_file) if args.split_file else None)


if __name__ == '__main__':
    main()
