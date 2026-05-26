#!/usr/bin/env python3
"""Preprocess Drive&Act RGB dataset into individual clip videos.

Input:  data/DriveAct_data/drive-and-act-rgb.zip  (or already extracted)
Output: data/DriveAct_data/clips/{activity}/{participant}_{run}_{start}_{end}.mp4
        data/DriveAct_data/splits/train.csv  (video_path, label, label_id, split)
        data/DriveAct_data/splits/val.csv

Uses midlevel annotations (34 activity classes), camera view: kinect_color.
Split 0 from the dataset's official 3-fold splits.
"""
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict


CAMERA = "kinect_color"
ANNOTATION_LEVEL = "midlevel"
SPLIT_IDX = 0          # use split_0
CAMERA_VIEW = "kinect_color"   # matches kinect_color video files directly


def extract_zip(zip_path: Path, out_dir: Path):
    if (out_dir / "drive_and_act").exists():
        print(f"[preprocess] Already extracted to {out_dir}")
        return
    print(f"[preprocess] Extracting {zip_path} → {out_dir} ...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(out_dir)], check=True)
    print("[preprocess] Extraction done")


def get_video_fps(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    try:
        num, den = result.stdout.strip().split("/")
        return float(num) / float(den)
    except Exception:
        return 30.0


def extract_clip(src: Path, dst: Path, frame_start: int, frame_end: int, fps: float):
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    t_start = frame_start / fps
    duration = (frame_end - frame_start) / fps
    if duration < 0.5:
        return False
    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t_start:.4f}", "-i", str(src),
        "-t", f"{duration:.4f}",
        "-vf", "scale=224:224",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an", str(dst)
    ], capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/DriveAct_data")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Limit clips for testing (default: all)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.data_root)
    zip_path = root / "drive-and-act-rgb.zip"
    extract_dir = root

    # Extract if needed
    if zip_path.exists():
        extract_zip(zip_path, extract_dir)

    ann_base = root / "drive_and_act" / "content" / "drive_and_act" / "activities_3s" / CAMERA_VIEW
    video_base = root / "drive_and_act" / "content" / "drive_and_act" / CAMERA

    if not ann_base.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ann_base}")
    if not video_base.exists():
        raise FileNotFoundError(f"Video dir not found: {video_base}")

    clips_dir = root / "clips"
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load train/val annotations
    rows = {}
    for split in ("train", "val"):
        csv_path = ann_base / f"{ANNOTATION_LEVEL}.chunks_90.split_{SPLIT_IDX}.{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation not found: {csv_path}")
        df = pd.read_csv(csv_path)
        rows[split] = df
        print(f"[preprocess] {split}: {len(df)} annotations, "
              f"{df['activity'].nunique()} activities")

    # Build class mapping
    all_activities = sorted(set(
        rows["train"]["activity"].unique().tolist() +
        rows["val"]["activity"].unique().tolist()
    ))
    act_to_id = {a: i for i, a in enumerate(all_activities)}
    print(f"[preprocess] Total activities: {len(all_activities)}")
    print(f"  {all_activities[:5]} ...")

    # FPS cache per video
    fps_cache: dict[str, float] = {}

    def get_fps(file_id: str) -> float:
        base = file_id.rsplit(".", 1)[0]  # strip camera suffix
        vid = video_base / f"{base}.{CAMERA}.mp4"
        if str(vid) not in fps_cache:
            fps_cache[str(vid)] = get_video_fps(vid) if vid.exists() else 30.0
        return fps_cache[str(vid)]

    output_rows = defaultdict(list)
    total = sum(len(df) for df in rows.values())
    done = 0

    for split, df in rows.items():
        if args.max_clips:
            df = df.head(args.max_clips)
        for _, row in df.iterrows():
            file_id = row["file_id"]               # e.g. "vp1/run1b_....ids_2"
            base = file_id.rsplit(".", 1)[0]       # "vp1/run1b_..."
            activity = row["activity"]
            fs = int(row["frame_start"])
            fe = int(row["frame_end"])

            src_video = video_base / f"{base}.{CAMERA}.mp4"
            if not src_video.exists():
                done += 1
                continue

            fps = get_fps(file_id)
            clip_name = f"{base.replace('/', '_')}_{fs}_{fe}.mp4"
            dst_clip = clips_dir / activity / clip_name

            ok = extract_clip(src_video, dst_clip, fs, fe, fps)
            if ok:
                output_rows[split].append({
                    "video_path": str(dst_clip),
                    "label": activity,
                    "label_id": act_to_id[activity],
                    "split": split,
                })

            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} clips processed")

    # Save splits
    for split, rlist in output_rows.items():
        out_df = pd.DataFrame(rlist)
        out_path = splits_dir / f"{split}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[preprocess] Saved {split}: {len(out_df)} clips → {out_path}")

    # Save class list
    pd.DataFrame({"label": all_activities, "label_id": range(len(all_activities))}).to_csv(
        splits_dir / "classes.csv", index=False
    )
    print(f"[preprocess] Done. {len(all_activities)} classes, "
          f"clips at {clips_dir}")


if __name__ == "__main__":
    main()
