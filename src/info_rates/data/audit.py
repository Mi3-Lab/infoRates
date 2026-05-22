"""Dataset auditing utilities for ACCV 2026 experiments."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import pandas as pd


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}


@dataclass
class VideoProbe:
    video_path: str
    exists: bool
    readable: bool
    fps: float
    num_frames: int
    duration: float
    width: int
    height: int
    error: str = ""


def probe_video(video_path: str | Path) -> VideoProbe:
    """Probe basic video metadata using OpenCV."""
    path = Path(video_path)
    if not path.exists():
        return VideoProbe(str(path), False, False, 0.0, 0, 0.0, 0, 0, "missing")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return VideoProbe(str(path), True, False, 0.0, 0, 0.0, 0, 0, "cannot_open")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = float(num_frames / fps) if fps > 0 else 0.0

    readable = False
    error = ""
    try:
        readable, _ = cap.read()
        if not readable:
            error = "no_decodable_frame"
    except Exception as exc:  # pragma: no cover - defensive around codec stack
        readable = False
        error = str(exc)
    finally:
        cap.release()

    return VideoProbe(
        str(path),
        True,
        bool(readable),
        fps,
        num_frames,
        duration,
        width,
        height,
        error,
    )


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_something_split(data_root: str | Path, split: str) -> pd.DataFrame:
    """Load a Something-Something V2 split and normalize labels."""
    data_root = Path(data_root)
    split_path = data_root / "labels" / f"{split}.json"
    records = load_json(split_path)
    df = pd.DataFrame(records)
    if "id" not in df.columns:
        raise ValueError(f"{split_path} does not contain an 'id' column")

    if "template" in df.columns:
        df["label"] = df["template"].astype(str)
    elif "label" not in df.columns:
        df["label"] = ""

    df["video_id"] = df["id"].astype(str)
    df["video_path"] = df["video_id"].map(
        lambda video_id: str(data_root / "videos" / f"{video_id}.webm")
    )
    df["dataset"] = "somethingv2"
    df["split"] = split
    return df[["dataset", "split", "video_id", "video_path", "label"]]


def load_something_label_map(data_root: str | Path) -> dict[str, int]:
    labels = load_json(Path(data_root) / "labels" / "labels.json")
    if isinstance(labels, dict):
        return {str(k): int(v) for k, v in labels.items()}
    return {str(label): idx for idx, label in enumerate(labels)}


def load_diving48_pose_manifest(
    pkl_path: str | Path,
    video_root: str | Path,
) -> pd.DataFrame:
    """Build a Diving48 video manifest from an OpenMMLab/PYSKL annotation pkl."""
    pkl_path = Path(pkl_path)
    video_root = Path(video_root)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    split_by_video = {}
    for split, names in data["split"].items():
        for name in names:
            split_by_video[str(name)] = str(split)

    rows = []
    for item in data["annotations"]:
        video_id = str(item["frame_dir"])
        label_id = int(item["label"])
        rows.append(
            {
                "dataset": "diving48",
                "split": split_by_video.get(video_id, "unknown"),
                "video_id": video_id,
                "video_path": str(video_root / f"{video_id}.mp4"),
                "label": f"class_{label_id:02d}",
                "label_id": label_id,
                "total_frames_annotation": int(item.get("total_frames", 0)),
            }
        )
    return pd.DataFrame(rows)


def normalize_something_label(label: str) -> str:
    return str(label).replace("[", "").replace("]", "").strip()


def add_label_ids(df: pd.DataFrame, label_map: dict[str, int]) -> pd.DataFrame:
    normalized_map = {normalize_something_label(k): v for k, v in label_map.items()}
    out = df.copy()
    out["label_clean"] = out["label"].map(normalize_something_label)
    out["label_id"] = out["label_clean"].map(normalized_map)
    return out


def attach_video_metadata(df: pd.DataFrame, probe_limit: int | None = None) -> pd.DataFrame:
    """Attach metadata columns to a manifest-like DataFrame."""
    rows = []
    iterator = df.itertuples(index=False)
    for index, row in enumerate(iterator):
        if probe_limit is not None and index >= probe_limit:
            break
        record = row._asdict()
        probe = probe_video(record["video_path"])
        record.update(asdict(probe))
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_manifest(df: pd.DataFrame) -> dict:
    summary = {
        "total_rows": int(len(df)),
        "existing_files": int(df["exists"].sum()) if "exists" in df else None,
        "readable_files": int(df["readable"].sum()) if "readable" in df else None,
        "missing_files": int((~df["exists"]).sum()) if "exists" in df else None,
        "splits": {},
        "labels": {},
    }
    if "split" in df:
        summary["splits"] = {str(k): int(v) for k, v in Counter(df["split"]).items()}
    if "label" in df:
        summary["labels"] = {
            "num_classes": int(df["label"].nunique()),
            "min_per_class": int(df.groupby("label").size().min()) if len(df) else 0,
            "max_per_class": int(df.groupby("label").size().max()) if len(df) else 0,
        }
    if "fps" in df and len(df):
        valid_fps = df.loc[df["fps"] > 0, "fps"]
        summary["fps"] = {
            "mean": float(valid_fps.mean()) if len(valid_fps) else 0.0,
            "min": float(valid_fps.min()) if len(valid_fps) else 0.0,
            "max": float(valid_fps.max()) if len(valid_fps) else 0.0,
        }
    return summary


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_manifest(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def class_balanced_subset(
    df: pd.DataFrame,
    samples_per_class: int,
    seed: int = 42,
    require_readable: bool = True,
) -> pd.DataFrame:
    """Create a class-balanced subset from a manifest."""
    if require_readable and "readable" in df:
        df = df[df["readable"]].copy()
    if "label" not in df:
        raise ValueError("class_balanced_subset requires a 'label' column")
    parts = []
    for _, group in df.groupby("label", sort=False):
        n = min(len(group), samples_per_class)
        parts.append(group.sample(n, random_state=seed))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def iter_videos(root: str | Path, extensions: Sequence[str] = tuple(VIDEO_EXTENSIONS)) -> Iterable[Path]:
    root = Path(root)
    allowed = {ext.lower() for ext in extensions}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path
