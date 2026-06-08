#!/usr/bin/env python3
"""
Convert UCFCrime frame-level PNGs into proper MP4 video clips.

UCFCrime frames are named {VideoID}_{FrameNum}.png — sequential frames
from real surveillance videos. We re-encode them as MP4 at 10fps.

FLAME is skipped — it is an aerial image classification dataset with
no temporal sequences; it is not suitable for video temporal aliasing research.
"""
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_ufc_crime_videos(fps: int = 10):
    raw_path = ROOT / "data/UCFCrime_data/raw_archives"
    out_base  = ROOT / "data/UCFCrime_data/videos"
    pattern   = re.compile(r'^(.+)_(\d+)\.png$')

    # Group frames by (split, label, video_id)
    groups = defaultdict(list)
    for split_dir in raw_path.glob("*/"):
        split = split_dir.name  # "Train" or "Test"
        for class_dir in split_dir.glob("*/"):
            label = class_dir.name
            for img in class_dir.glob("*.png"):
                m = pattern.match(img.name)
                if m:
                    groups[(split, label, m.group(1))].append((int(m.group(2)), img))

    total   = len(groups)
    done    = 0
    errors  = 0
    skipped = 0

    print(f"Total video clips to create: {total}")
    for (split, label, video_id), frames in groups.items():
        out_dir = out_base / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"{video_id}.mp4"

        if out_mp4.exists():
            skipped += 1
            continue

        frames_sorted = [p for _, p in sorted(frames)]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i, src in enumerate(frames_sorted):
                shutil.copy2(src, tmpdir / f"frame_{i:06d}.png")

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(tmpdir / "frame_%06d.png"),
                "-vf", "scale=320:240",
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                str(out_mp4),
            ]
            ret = subprocess.run(cmd, capture_output=True)

        if ret.returncode == 0:
            done += 1
        else:
            errors += 1
            print(f"  ❌ {video_id}: {ret.stderr.decode()[-100:]}")

        total_processed = done + errors + skipped
        if total_processed % 50 == 0 or total_processed == total:
            print(f"  [{total_processed}/{total}] done={done} skipped={skipped} errors={errors}")

    print(f"\n✅ UCFCrime: {done} new + {skipped} existing = {done+skipped}/{total} videos ({errors} errors)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("Building UCF-Crime video clips from PNG frames")
    print("FLAME skipped (image dataset, no temporal sequences)")
    print("=" * 60)
    build_ufc_crime_videos(fps=args.fps)
