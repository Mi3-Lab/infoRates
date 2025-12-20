import os
import subprocess
from glob import glob
from typing import List, Tuple
import random
from pathlib import Path
import av
import pandas as pd


def extract_ucf101(rar_file: str, split_zip: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(rar_file):
        subprocess.run(["unrar", "x", rar_file, out_dir])
    else:
        print(f"UCF101 rar not found: {rar_file}")
    if os.path.exists(split_zip):
        subprocess.run(["unzip", "-q", split_zip, "-d", out_dir])
    else:
        print(f"Split zip not found: {split_zip}")


def list_classes(video_root: str) -> List[str]:
    return sorted([d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))])


def list_videos(video_root: str) -> List[str]:
    return sorted(glob(os.path.join(video_root, "*", "*.avi")))


def train_val_test_split(video_root: str, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    all_videos = list_videos(video_root)
    random.seed(seed)
    random.shuffle(all_videos)
    n = len(all_videos)
    n_train = int(train_ratio * n)
    n_val = int((train_ratio + val_ratio) * n)
    train_files = all_videos[:n_train]
    val_files = all_videos[n_train:n_val]
    test_files = all_videos[n_val:]
    return train_files, val_files, test_files


def split_video_fixed(video_path: str, out_dir: Path, target_frames: int = 50) -> List[Path]:
    container = av.open(video_path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    n_frames = len(frames)
    segments = n_frames // target_frames
    saved: List[Path] = []

    label = Path(video_path).parent.name
    out_label_dir = out_dir / label
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for i in range(segments):
        seg_frames = frames[i * target_frames:(i + 1) * target_frames]
        seg_path = out_label_dir / f"{Path(video_path).stem}_seg{i:02d}.mp4"
        out = av.open(str(seg_path), "w")
        stream = out.add_stream("mpeg4", rate=25)
        stream.width, stream.height = seg_frames[0].shape[1], seg_frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for frame_array in seg_frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
        out.close()
        saved.append(seg_path)
    return saved


def _process_single_video(args):
    """Helper function for parallel processing."""
    video_path, out_dir, target_frames = args
    try:
        segs = split_video_fixed(video_path, out_dir, target_frames)
        label = Path(video_path).parent.name
        return [{"video_path": str(s), "label": label} for s in segs]
    except Exception as e:
        print(f"⚠️ {video_path} -> {e}")
        return []


def build_fixed_manifest(video_paths: List[str], out_dir: Path, target_frames: int = 50, workers: int = None) -> pd.DataFrame:
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cores free
    
    print(f"🎬 Processing {len(video_paths)} videos into {target_frames}-frame clips...")
    print(f"⚡ Using {workers} parallel workers")
    
    manifest = []
    args_list = [(vp, out_dir, target_frames) for vp in video_paths]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=len(video_paths), desc="Splitting videos") as pbar:
            for result in executor.map(_process_single_video, args_list, chunksize=10):
                manifest.extend(result)
                pbar.update(1)
    
    return pd.DataFrame(manifest)
