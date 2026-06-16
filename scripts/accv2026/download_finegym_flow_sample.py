#!/usr/bin/env python3
"""
Targeted FineGym download for the E3 spectral (optical-flow) analysis.

Unlike download_finegym.py (which downloads the full train+val split,
~29k clips), this script downloads only the specific val clips needed
by e3_spectral_analysis.py (5 videos/class x 97 classes = ~471 clips,
31 unique YouTube source videos), read from
/tmp/finegym_needed_clips.csv (columns: video_id, label_id).
"""
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "data/FineGym_data/annotations"
VID_DIR = ROOT / "data/FineGym_data/videos"
RAW_DIR = ROOT / "data/FineGym_data/raw_events"

YT_BASE = "https://www.youtube.com/watch?v="
NEEDED_CSV = Path("/tmp/finegym_needed_clips.csv")


def parse_clip_id(clip_id):
    m = re.match(r'^(.+)_E_(\d+)_(\d+)_A_(\d+)_(\d+)$', clip_id)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)


def get_event_timestamps(ann, yt_id, ev_id):
    ev = ann.get(yt_id, {}).get(ev_id)
    if ev and ev.get('timestamps'):
        return ev['timestamps'][0]
    return None


def get_subaction_timestamps(ann, yt_id, ev_id, act_id):
    ev = ann.get(yt_id, {}).get(ev_id)
    if not ev or not ev.get('segments'):
        return None
    act = ev['segments'].get(act_id)
    if not act or not act.get('timestamps'):
        return None
    all_ts = act['timestamps']
    return all_ts[0][0], all_ts[-1][1]


def download_event(yt_id, ev_start, ev_end, out_path, quality="360p"):
    out_path = Path(out_path)
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dl_start = max(0, ev_start - 1)
    dl_end = ev_end + 1
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "--download-sections", f"*{dl_start:.2f}-{dl_end:.2f}",
        "--format", "bestvideo[height<=360][ext=mp4]+bestaudio/best[height<=360]/best",
        "--merge-output-format", "mp4",
        "--output", str(out_path),
        f"{YT_BASE}{yt_id}",
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, timeout=300)
        if ret.returncode != 0:
            print(f"    yt-dlp stderr: {ret.stderr.decode()[-300:]}")
        return ret.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout downloading {yt_id}")
        return False


def trim_clip(src_path, out_path, clip_start, clip_end):
    out_path = Path(out_path)
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = clip_end - clip_start
    cmd = [
        "ffmpeg", "-y", "-ss", str(clip_start), "-i", str(src_path),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-vf", "scale=320:240", "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ]
    ret = subprocess.run(cmd, capture_output=True, timeout=60)
    return ret.returncode == 0


def main():
    with open(ANN_DIR / "annotation.json") as f:
        ann = json.load(f)

    needed = pd.read_csv(NEEDED_CSV)
    print(f"Clips needed: {len(needed)}")

    by_event = defaultdict(list)
    for _, row in needed.iterrows():
        clip_id, label = row["video_id"], row["label_id"]
        parsed = parse_clip_id(clip_id)
        if not parsed:
            print(f"  skip (unparsable): {clip_id}")
            continue
        yt_id, ev_s, ev_e, act_s, act_e = parsed
        ev_id = f"E_{ev_s}_{ev_e}"
        act_id = f"A_{act_s}_{act_e}"
        by_event[(yt_id, ev_id)].append((clip_id, act_id, label))

    jobs = list(by_event.items())  # [((yt_id, ev_id), clips), ...]
    print(f"Unique events to fetch: {len(jobs)}  (Unique YouTube videos: {len(set(j[0][0] for j in jobs))})")

    done = skip = err = dl_err = 0
    lock = __import__("threading").Lock()

    def process_event(job):
        nonlocal done, skip, err, dl_err
        (yt_id, ev_id), clips = job
        ev_ts = get_event_timestamps(ann, yt_id, ev_id)
        if not ev_ts:
            with lock:
                err += len(clips)
            return f"  no timestamp for {yt_id}/{ev_id}"
        ev_start, ev_end = ev_ts
        raw_mp4 = RAW_DIR / yt_id / f"{ev_id}.mp4"
        if not raw_mp4.exists():
            ok = download_event(yt_id, ev_start, ev_end, raw_mp4)
            if not ok:
                with lock:
                    dl_err += 1
                    err += len(clips)
                return f"  download failed: {yt_id}/{ev_id}"
        local_done = local_skip = local_err = 0
        for clip_id, act_id, label in clips:
            out_path = VID_DIR / "val" / str(label) / f"{clip_id}.mp4"
            if out_path.exists():
                local_skip += 1
                continue
            act_ts = get_subaction_timestamps(ann, yt_id, ev_id, act_id)
            if not act_ts:
                act_ts = (0.0, ev_end - ev_start)
            buf = 1.0
            clip_start = buf + act_ts[0]
            clip_end = buf + act_ts[1]
            ok = trim_clip(raw_mp4, out_path, clip_start, clip_end)
            if ok:
                local_done += 1
            else:
                local_err += 1
        with lock:
            done += local_done
            skip += local_skip
            err += local_err
        return f"  {yt_id}/{ev_id}: done={local_done} skip={local_skip} err={local_err}  [total done={done} skip={skip} err={err} dl_err={dl_err}]"

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process_event, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures)):
            print(f"[{i+1}/{len(futures)}] {fut.result()}")

    print(f"\nDone: {done} new, {skip} existing, {err} errors ({dl_err} download failures)")


if __name__ == "__main__":
    main()
