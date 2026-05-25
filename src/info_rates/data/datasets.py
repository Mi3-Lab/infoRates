"""Unified dataset loader for ACCV 2026 multi-dataset experiments.

Returns (class_names, train_files, val_files) where *_files is a list of
(absolute_video_path, label_id) tuples — compatible with UCFDataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List


DataSplit = List[Tuple[str, int]]


def load_ucf101(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """UCF101 split-1: trainlist01.txt for train, testlist01.txt as val."""
    root = Path(data_root)
    splits_dir = root / "ucfTrainTestlist"
    video_root = root / "UCF-101"

    class_mapping: dict[str, int] = {}
    with open(splits_dir / "classInd.txt") as f:
        for line in f:
            idx, name = line.strip().split(" ", 1)
            class_mapping[name] = int(idx) - 1  # 1-indexed → 0-indexed

    class_names = [n for n, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

    train_files: DataSplit = []
    with open(splits_dir / "trainlist01.txt") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                rel_path, label_str = parts[0], parts[1]
                train_files.append((str(video_root / rel_path), int(label_str) - 1))

    val_files: DataSplit = []
    with open(splits_dir / "testlist01.txt") as f:
        for line in f:
            rel_path = line.strip()
            if rel_path:
                class_name = Path(rel_path).parent.name
                label_id = class_mapping.get(class_name, -1)
                if label_id >= 0:
                    val_files.append((str(video_root / rel_path), label_id))

    return class_names, train_files, val_files


def load_hmdb51(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """HMDB51 using pre-built train/val split CSVs in data_root/splits/."""
    import pandas as pd

    root = Path(data_root)
    train_df = pd.read_csv(root / "splits" / "train.csv")
    val_df = pd.read_csv(root / "splits" / "val.csv")

    # Sort unique labels by label_id for consistent class list
    classes_df = (
        train_df[["label", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")
    )
    class_names = classes_df["label"].tolist()

    train_files: DataSplit = [
        (str(Path(r["video_path"])), int(r["label_id"]))
        for _, r in train_df.iterrows()
    ]
    val_files: DataSplit = [
        (str(Path(r["video_path"])), int(r["label_id"]))
        for _, r in val_df.iterrows()
    ]
    return class_names, train_files, val_files


def load_diving48(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """Diving48 from the pre-built manifest CSV (evaluations/accv2026/manifests/)."""
    import pandas as pd

    manifest_path = Path("evaluations/accv2026/manifests/diving48_manifest.csv")
    if not manifest_path.exists():
        # fallback relative to data_root
        manifest_path = Path(data_root) / "annotations" / "diving48_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Diving48 manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    num_classes = int(df["label_id"].max()) + 1
    class_names = [f"class_{i:02d}" for i in range(num_classes)]

    def _rows(split_name: str) -> DataSplit:
        sub = df[df["split"] == split_name].copy()
        if "exists" in sub.columns:
            sub = sub[sub["exists"].astype(bool)]
        return [(str(Path(r["video_path"])), int(r["label_id"])) for _, r in sub.iterrows()]

    return class_names, _rows("train"), _rows("test")


def load_wlasl(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """WLASL-2000 from WLASL_v0.3.json + downloaded raw_videos/."""
    import json
    from urllib.parse import urlparse, parse_qs

    root = Path(data_root)
    json_path = root / "WLASL_v0.3.json"
    videos_dir = root / "raw_videos"

    if not json_path.exists():
        raise FileNotFoundError(f"WLASL JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    # Build lookup: filename stem → full path
    available: dict[str, str] = {
        p.stem: str(p) for p in videos_dir.glob("*.mp4")
    }

    def _resolve(video_id: str, url: str) -> str | None:
        if video_id in available:
            return available[video_id]
        if "youtube" in url or "youtu.be" in url:
            parsed = urlparse(url)
            yt_id = parse_qs(parsed.query).get("v", [None])[0]
            if yt_id and yt_id in available:
                return available[yt_id]
        return None

    train_files: DataSplit = []
    val_files: DataSplit = []

    # Build compact class list: only glosses with at least one available file
    gloss_to_id: dict[str, int] = {}
    class_names: List[str] = []
    cid = 0

    for entry in data:
        gloss = entry["gloss"]
        has_any = any(
            _resolve(inst["video_id"], inst.get("url", "")) is not None
            for inst in entry["instances"]
        )
        if not has_any:
            continue
        if gloss not in gloss_to_id:
            gloss_to_id[gloss] = cid
            class_names.append(gloss)
            cid += 1

    for entry in data:
        gloss = entry["gloss"]
        if gloss not in gloss_to_id:
            continue
        label_id = gloss_to_id[gloss]
        for inst in entry["instances"]:
            path = _resolve(inst["video_id"], inst.get("url", ""))
            if path is None:
                continue
            split = inst.get("split", "train")
            if split == "train":
                train_files.append((path, label_id))
            elif split in ("val", "test"):
                val_files.append((path, label_id))

    return class_names, train_files, val_files


def load_epic_kitchens(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """EPIC-Kitchens 100 action recognition using verb classes (97 classes).

    Expects pre-extracted clips at {data_root}/clips/{participant_id}/{narration_id}.mp4
    and annotation CSVs at {data_root}/annotations/EPIC_100_{train,validation}.csv.
    Only includes rows for which a clip file actually exists (lightly-ai subset).
    """
    import pandas as pd

    root = Path(data_root)
    clips_dir = root / "clips"
    ann_dir = root / "annotations"

    train_df = pd.read_csv(ann_dir / "EPIC_100_train.csv")
    val_df = pd.read_csv(ann_dir / "EPIC_100_validation.csv")

    # Load verb class names
    verb_df = pd.read_csv(ann_dir / "EPIC_100_verb_classes.csv")
    # columns: id, key, instances (semi-colon separated synonyms)
    class_names = [""] * len(verb_df)
    for _, row in verb_df.iterrows():
        class_names[int(row["id"])] = str(row["key"])

    def _build_split(df: "pd.DataFrame") -> DataSplit:
        result: DataSplit = []
        for _, row in df.iterrows():
            pid = row["participant_id"]
            nid = row["narration_id"]
            clip_path = clips_dir / pid / f"{nid}.mp4"
            if clip_path.exists():
                result.append((str(clip_path), int(row["verb_class"])))
        return result

    return class_names, _build_split(train_df), _build_split(val_df)


_LOADERS = {
    "ucf101": (load_ucf101, "data/UCF101_data"),
    "hmdb51": (load_hmdb51, "data/HMDB51_data"),
    "diving48": (load_diving48, "data/Diving48_data"),
    "wlasl": (load_wlasl, "data/WLASL_data"),
    "epic_kitchens": (load_epic_kitchens, "data/EPIC_data"),
}


def load_dataset(
    name: str, data_root: str | None = None
) -> Tuple[List[str], DataSplit, DataSplit]:
    """Dispatch by dataset name. data_root overrides the default path."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_LOADERS)}")
    loader_fn, default_root = _LOADERS[name]
    return loader_fn(data_root or default_root)


def build_eval_manifest(
    name: str,
    data_root: str | None = None,
    split: str = "val",
    samples_per_class: int = 20,
    seed: int = 42,
) -> "pd.DataFrame":
    """Build a fixed-budget eval manifest (video_path, label_id, ...) for any dataset."""
    import pandas as pd
    import random

    _, train_files, val_files = load_dataset(name, data_root)
    files = val_files  # use val/test split for evaluation

    # Group by label_id and sample up to samples_per_class
    from collections import defaultdict
    by_label: dict[int, list] = defaultdict(list)
    for path, label_id in files:
        by_label[label_id].append(path)

    rng = random.Random(seed)
    rows = []
    for label_id, paths in sorted(by_label.items()):
        selected = rng.sample(paths, min(len(paths), samples_per_class))
        for path in selected:
            rows.append({
                "dataset": name,
                "split": split,
                "video_id": Path(path).stem,
                "video_path": path,
                "label": f"class_{label_id:02d}",
                "label_clean": f"class_{label_id:02d}",
                "label_id": label_id,
                "exists": Path(path).exists(),
            })

    return pd.DataFrame(rows)
