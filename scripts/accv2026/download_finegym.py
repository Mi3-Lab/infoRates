#!/usr/bin/env python3
"""
Download FineGym (Gym99) dataset from YouTube.

Strategy:
1. Parse gym99_train.txt + gym99_val.txt to find all needed (yt_id, event) pairs
2. For each YouTube video, download only the needed event sections (not full video)
3. Trim each sub-action clip from the downloaded event segment
4. Save clips as {clip_id}.mp4 in data/FineGym_data/videos/{split}/{label}/
5. Create manifest CSV for training
"""
import json
import re
import subprocess
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR  = ROOT / "data/FineGym_data/annotations"
VID_DIR  = ROOT / "data/FineGym_data/videos"
RAW_DIR  = ROOT / "data/FineGym_data/raw_events"   # temp event downloads

YT_BASE  = "https://www.youtube.com/watch?v="
FPS      = 25   # gymnastics broadcast standard


def parse_split(txt_path):
    """Return list of (clip_id, label_int) from gym99 split file."""
    entries = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            entries.append((parts[0], int(parts[1])))
    return entries


def parse_clip_id(clip_id):
    """
    clip_id: {yt_id}_E_{ev_s}_{ev_e}_A_{act_s}_{act_e}
    Returns (yt_id, ev_s, ev_e, act_s, act_e) as strings.
    """
    m = re.match(r'^(.+)_E_(\d+)_(\d+)_A_(\d+)_(\d+)$', clip_id)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)


def get_event_timestamps(ann, yt_id, ev_id):
    """Return (start_sec, end_sec) of event in the YouTube video."""
    ev = ann.get(yt_id, {}).get(ev_id)
    if ev and ev.get('timestamps'):
        return ev['timestamps'][0]
    return None


def get_subaction_timestamps(ann, yt_id, ev_id, act_id):
    """Return (start_sec, end_sec) of sub-action RELATIVE to event start."""
    ev = ann.get(yt_id, {}).get(ev_id)
    if not ev or not ev.get('segments'):
        return None
    act = ev['segments'].get(act_id)
    if not act or not act.get('timestamps'):
        return None
    all_ts = act['timestamps']
    return all_ts[0][0], all_ts[-1][1]


def download_event(yt_id, ev_start, ev_end, out_path, quality="360p"):
    """Download a specific time range from YouTube using yt-dlp."""
    out_path = Path(out_path)
    if out_path.exists():
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add 1s buffer on each side
    dl_start = max(0, ev_start - 1)
    dl_end   = ev_end + 1

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--download-sections", f"*{dl_start:.2f}-{dl_end:.2f}",
        "--format", "bestvideo[height<=360][ext=mp4]+bestaudio/best[height<=360]/best",
        "--merge-output-format", "mp4",
        "--output", str(out_path),
        f"{YT_BASE}{yt_id}",
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, timeout=300)
        return ret.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⏱ Timeout downloading {yt_id}/{out_path.name}")
        return False


def trim_clip(src_path, out_path, clip_start, clip_end):
    """Trim clip from src_path using ffmpeg."""
    out_path = Path(out_path)
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = clip_end - clip_start

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-i", str(src_path),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-vf", "scale=320:240",
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_path),
    ]
    ret = subprocess.run(cmd, capture_output=True, timeout=60)
    return ret.returncode == 0


def load_categories():
    cat_map = {}
    with open(ANN_DIR / "gym99_categories.txt") as f:
        for line in f:
            m = re.match(r'Clabel:\s*(\d+);.*', line)
            if m:
                cat_map[int(m.group(1))] = line.strip()
    return cat_map


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Limit number of YouTube videos to download (0=all)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("Downloading FineGym (Gym99) from YouTube")
    print("=" * 60)

    # Load annotations
    with open(ANN_DIR / "annotation.json") as f:
        ann = json.load(f)

    # Parse train + val splits
    train_entries = parse_split(ANN_DIR / "gym99_train.txt")
    val_entries   = parse_split(ANN_DIR / "gym99_val.txt")
    all_entries = [("train", cid, lbl) for cid, lbl in train_entries] + \
                  [("val",   cid, lbl) for cid, lbl in val_entries]

    print(f"  Train clips: {len(train_entries)}")
    print(f"  Val clips:   {len(val_entries)}")
    print(f"  Total clips: {len(all_entries)}")

    # Group by (yt_id, ev_id) to batch downloads
    # key: (yt_id, ev_id) → list of (split, clip_id, act_id, label)
    by_event = defaultdict(list)
    skipped_parse = 0
    for split, clip_id, label in all_entries:
        parsed = parse_clip_id(clip_id)
        if not parsed:
            skipped_parse += 1
            continue
        yt_id, ev_s, ev_e, act_s, act_e = parsed
        ev_id  = f"E_{ev_s}_{ev_e}"
        act_id = f"A_{act_s}_{act_e}"
        by_event[(yt_id, ev_id)].append((split, clip_id, act_id, label))

    print(f"  Unique events to download: {len(by_event)}")
    print(f"  Unique YouTube videos: {len(set(k[0] for k in by_event))}")
    if skipped_parse:
        print(f"  Parse errors: {skipped_parse}")

    # Process events
    done_clips = 0
    skip_clips = 0
    error_clips = 0
    error_dl = 0

    by_yt = defaultdict(list)
    for (yt_id, ev_id), clips in by_event.items():
        by_yt[yt_id].append((ev_id, clips))

    yt_ids = list(by_yt.keys())
    if args.max_videos:
        yt_ids = yt_ids[:args.max_videos]
        print(f"\n  (Limited to {args.max_videos} YouTube videos)")

    for i, yt_id in enumerate(yt_ids):
        print(f"\n[{i+1}/{len(yt_ids)}] YouTube: {yt_id}")

        for ev_id, clips in by_yt[yt_id]:
            ev_ts = get_event_timestamps(ann, yt_id, ev_id)
            if not ev_ts:
                print(f"  ⚠ No timestamp for {ev_id}")
                error_clips += len(clips)
                continue

            ev_start, ev_end = ev_ts
            raw_mp4 = RAW_DIR / yt_id / f"{ev_id}.mp4"

            # Download event if needed
            if not raw_mp4.exists():
                ok = download_event(yt_id, ev_start, ev_end, raw_mp4)
                if not ok:
                    print(f"  ❌ Download failed: {yt_id}/{ev_id}")
                    error_dl += 1
                    error_clips += len(clips)
                    continue

            # Trim each sub-action clip
            for split, clip_id, act_id, label in clips:
                out_path = VID_DIR / split / str(label) / f"{clip_id}.mp4"
                if out_path.exists():
                    skip_clips += 1
                    continue

                act_ts = get_subaction_timestamps(ann, yt_id, ev_id, act_id)
                if not act_ts:
                    # Fall back to full event
                    act_ts = (0.0, ev_end - ev_start)

                # Absolute position within the downloaded segment
                # (yt-dlp adds 1s buffer before ev_start)
                buf = 1.0
                clip_start = buf + act_ts[0]
                clip_end   = buf + act_ts[1]

                ok = trim_clip(raw_mp4, out_path, clip_start, clip_end)
                if ok:
                    done_clips += 1
                else:
                    error_clips += 1

        total = done_clips + skip_clips + error_clips
        print(f"  Progress: done={done_clips} skip={skip_clips} err={error_clips} dl_err={error_dl}")

    print("\n" + "=" * 60)
    print(f"✅ FineGym download complete!")
    print(f"   Done: {done_clips} new clips")
    print(f"   Skip: {skip_clips} existing")
    print(f"   Errors: {error_clips} clips, {error_dl} download failures")
    print(f"   Output: {VID_DIR}")


if __name__ == "__main__":
    main()
