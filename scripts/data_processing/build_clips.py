#!/usr/bin/env python3
"""Build fixed-length clips and manifest from raw UCF101 videos."""
import argparse
from pathlib import Path
import pandas as pd
from info_rates.data.ucf101 import list_videos, build_fixed_manifest


def main():
    parser = argparse.ArgumentParser(description="Build fixed-frame clips and manifest from UCF101 videos.")
    parser.add_argument("--video-root", required=True, help="Root folder of UCF101 videos (e.g., UCF101_data/UCF-101)")
    parser.add_argument("--out", default="data/UCF101_data/UCF101_50f", help="Output directory for 50-frame clips")
    parser.add_argument("--frames", type=int, default=50, help="Target frames per segment")
    parser.add_argument("--manifest", default="data/UCF101_data/manifests/ucf101_50f.csv", help="Path to save manifest CSV")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 2)")
    args = parser.parse_args()

    video_paths = list_videos(args.video_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    df = build_fixed_manifest(video_paths, out_dir, target_frames=args.frames, workers=args.workers)
    df.to_csv(args.manifest, index=False)
    print(f"✅ Created {len(df)} clips across {df['label'].nunique()} classes.")


if __name__ == "__main__":
    main()
