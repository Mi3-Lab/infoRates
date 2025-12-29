import numpy as np
import cv2
from decord import VideoReader, cpu
from typing import Tuple, List, Optional, Dict


def extract_uniform_frames(video_path: str, num_frames: int = 32, resize: int = 224) -> np.ndarray:
    """
    Uniformly extracts `num_frames` frames across the full video.
    Ensures consistent temporal coverage for aliasing studies.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    idx = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    frames = vr.get_batch(idx).asnumpy()
    frames = np.array([cv2.resize(f, (resize, resize)) for f in frames])
    return frames


def apply_aliasing(frames: np.ndarray, frame_percent: int = 100, stride: int = 1) -> np.ndarray:
    """
    Simulates temporal aliasing by reducing coverage and increasing stride.
    - frame_percent: how much of the base clip to keep (e.g., 25, 50, 100)
    - stride: temporal spacing between sampled frames
    """
    n_total = len(frames)
    n_use = max(1, int(n_total * frame_percent / 100))
    sampled = frames[:n_use:stride]
    return sampled


def subsample(frames: np.ndarray, coverage: int = 100, stride: int = 1) -> np.ndarray:
    n_total = len(frames)
    n_keep = max(1, int(n_total * coverage / 100))
    return frames[:n_keep:stride]


def norm_label(s: str) -> str:
    return s.lower().replace("_", "").replace(" ", "")


def extract_and_prepare(row: Dict[str, str], cov: int, stride: int, resize: int = 224, num_select: int = 8) -> Tuple[Optional[List[np.ndarray]], Optional[str]]:
    """
    Decode, subsample, and select `num_select` frames for a single video specified by manifest row.
    Row must contain keys: "video_path" and "label".
    
    Note: num_select should match the model's frame requirement:
    - TimeSformer: 8 frames
    - VideoMAE: 16 frames
    - ViViT: 32 frames
    """
    try:
        vr = VideoReader(row["video_path"], ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            return None, None

        # Sample a modest number of frames uniformly to avoid decoding entire video (more robust)
        base_n = min(total_frames, max(num_select * 4, num_select))
        idx_base = np.linspace(0, max(total_frames - 1, 0), base_n).astype(int)
        # Read frames individually to avoid bulk decoding errors on some files
        frames_list = []
        for i in idx_base:
            try:
                f = vr.get_batch([int(i)]).asnumpy()[0]
                frames_list.append(f)
            except Exception:
                continue
        if len(frames_list) == 0:
            return None, None
        import numpy as _np
        frames = _np.stack(frames_list, axis=0)

        # Apply coverage and stride on the sampled base frames
        n_keep = max(1, int(len(frames) * cov / 100))
        frames = frames[:n_keep:stride]

        # Ensure we have at least num_select frames to sample from
        if len(frames) < num_select:
            frames = np.repeat(frames, (num_select // len(frames)) + 1, axis=0)[:num_select]

        idx = np.linspace(0, len(frames) - 1, num_select).astype(int)
        selected = [cv2.resize(frames[j], (resize, resize)) for j in idx]
        return selected, row["label"]
    except Exception:
        return None, None
