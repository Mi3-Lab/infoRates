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


def load_wlasl100(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """WLASL-100: first 100 glosses of WLASL_v0.3.json (same raw_videos as WLASL-2000).

    100 classes, ~1000 train / ~240 val videos available on disk.
    """
    import json
    from urllib.parse import urlparse, parse_qs

    root = Path(data_root)
    json_path = root / "WLASL_v0.3.json"
    videos_dir = root / "raw_videos"

    if not json_path.exists():
        raise FileNotFoundError(f"WLASL JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

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
    gloss_to_id: dict[str, int] = {}
    class_names: List[str] = []
    cid = 0

    for entry in data[:100]:  # first 100 glosses only
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

    for entry in data[:100]:
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


def load_epic_kitchens(data_root: str, val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[str], DataSplit, DataSplit]:
    """EPIC-Kitchens 100 action recognition using verb classes (97 classes).

    Expects pre-extracted clips at {data_root}/clips/{participant_id}/{narration_id}.mp4
    (lightly-ai/epic-kitchens-100-clips, extension portion only).

    The train/val split is frozen to a CSV the first time it is computed so that
    incremental clip downloads do not alter which clips belong to val (which would
    cause data leakage between subsequent training runs and evaluation).
    Split CSVs: {data_root}/splits/epic_train_split.csv and epic_val_split.csv.
    """
    import pandas as pd
    import random

    root = Path(data_root)
    clips_dir = root / "clips"
    ann_dir = root / "annotations"
    splits_dir = root / "splits"
    train_csv = splits_dir / "epic_train_split.csv"
    val_csv = splits_dir / "epic_val_split.csv"

    # Load verb class names (always needed)
    verb_df = pd.read_csv(ann_dir / "EPIC_100_verb_classes.csv")
    class_names = [""] * len(verb_df)
    for _, row in verb_df.iterrows():
        class_names[int(row["id"])] = str(row["key"])

    # Use frozen split if it already exists
    if train_csv.exists() and val_csv.exists():
        tr = pd.read_csv(train_csv)
        va = pd.read_csv(val_csv)
        train_files: DataSplit = list(zip(tr["path"].tolist(), tr["label_id"].tolist()))
        val_files: DataSplit = list(zip(va["path"].tolist(), va["label_id"].tolist()))
        # Filter to only existing clips (handles partial downloads without changing split)
        train_files = [(p, l) for p, l in train_files if Path(p).exists()]
        val_files = [(p, l) for p, l in val_files if Path(p).exists()]
        return class_names, train_files, val_files

    # First run: compute the split and freeze it to CSV
    train_ann = pd.read_csv(ann_dir / "EPIC_100_train.csv")
    all_clips: DataSplit = []
    for _, row in train_ann.iterrows():
        pid = row["participant_id"]
        nid = row["narration_id"]
        clip_path = clips_dir / pid / f"{nid}.mp4"
        if clip_path.exists():
            all_clips.append((str(clip_path), int(row["verb_class"])))

    from collections import defaultdict
    by_class: dict[int, list] = defaultdict(list)
    for path, label in all_clips:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    train_files = []
    val_files = []
    for label_id in sorted(by_class):
        items = by_class[label_id]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_files.extend(items[:n_val])
        train_files.extend(items[n_val:])

    # Persist the frozen split
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": [p for p, _ in train_files], "label_id": [l for _, l in train_files]}).to_csv(train_csv, index=False)
    pd.DataFrame({"path": [p for p, _ in val_files], "label_id": [l for _, l in val_files]}).to_csv(val_csv, index=False)

    return class_names, train_files, val_files


def load_autsl(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """AUTSL Turkish Sign Language — 226 classes, ~38k videos.

    Expects preprocessed splits at {data_root}/splits/train.csv and val.csv.
    Run scripts/accv2026/preprocess_autsl.py first.
    """
    import pandas as pd
    root = Path(data_root)
    train_df = pd.read_csv(root / "splits" / "train.csv")
    val_df   = pd.read_csv(root / "splits" / "val.csv")

    classes_df = (
        pd.concat([train_df[["label", "label_id"]], val_df[["label", "label_id"]]])
        .drop_duplicates().sort_values("label_id")
    )
    class_names = classes_df["label"].tolist()

    def _rows(df: "pd.DataFrame") -> DataSplit:
        return [(str(r["video_path"]), int(r["label_id"])) for _, r in df.iterrows()
                if Path(str(r["video_path"])).exists()]

    return class_names, _rows(train_df), _rows(val_df)


def load_driveact(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """Drive&Act in-cabin activity recognition — 34 midlevel classes.

    Expects preprocessed clips at {data_root}/splits/train.csv and val.csv.
    Run scripts/accv2026/preprocess_driveact.py first.
    """
    import pandas as pd
    root = Path(data_root)
    train_df = pd.read_csv(root / "splits" / "train.csv")
    val_df   = pd.read_csv(root / "splits" / "val.csv")

    classes_df = (
        pd.concat([train_df[["label", "label_id"]], val_df[["label", "label_id"]]])
        .drop_duplicates().sort_values("label_id")
    )
    class_names = classes_df["label"].tolist()

    def _rows(df: "pd.DataFrame") -> DataSplit:
        return [(str(r["video_path"]), int(r["label_id"])) for _, r in df.iterrows()
                if Path(str(r["video_path"])).exists()]

    return class_names, _rows(train_df), _rows(val_df)


def load_kinetics400(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """Kinetics-400 val set — 400 classes, ~19.8k videos (val only, no training split).

    Uses pre-trained VideoMAE checkpoint; this loader returns empty train_files.
    """
    import pandas as pd
    manifest = Path("evaluations/accv2026/manifests/kinetics400_val.csv")
    if not manifest.exists():
        manifest = Path(data_root) / "manifests" / "kinetics400_val.csv"
    df = pd.read_csv(manifest)

    # Load class names from HF config cache if available
    hf_id2label_path = Path(data_root) / "id2label.json"
    if hf_id2label_path.exists():
        import json
        id2label = json.load(open(hf_id2label_path))
    else:
        id2label = {str(i): f"class_{i:03d}" for i in range(400)}

    num_classes = int(df["label"].max()) + 1
    class_names = [id2label.get(str(i), f"class_{i:03d}") for i in range(num_classes)]

    val_files: DataSplit = [
        (str(r["video_path"]), int(r["label"]))
        for _, r in df.iterrows()
        if Path(str(r["video_path"])).exists()
    ]
    return class_names, [], val_files  # no train split available


def load_from_manifest(manifest_name: str, data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """Load dataset from pre-computed CSV manifest (for FLAME, UFC-Crime, etc.)."""
    import pandas as pd

    # Find manifest in repo root / evaluations/accv2026/manifests/
    root = Path(data_root).resolve()
    # Navigate up to repo root
    while root.name != "infoRates" and root.parent != root:
        root = root.parent
    manifest_path = root / "evaluations/accv2026/manifests" / f"{manifest_name}_full.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Extract unique class names and create label mapping
    unique_labels = sorted(df["label"].unique())
    class_names = list(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    # Separate train and validation splits
    train_files: DataSplit = [
        (str(r["video_path"]), label_to_id[r["label"]])
        for _, r in df.iterrows()
        if (r["split"] == "train" or r["split"] == "training") and Path(str(r["video_path"])).exists()
    ]

    val_files: DataSplit = [
        (str(r["video_path"]), label_to_id[r["label"]])
        for _, r in df.iterrows()
        if (r["split"] == "valid" or r["split"] == "validation") and Path(str(r["video_path"])).exists()
    ]

    # Fallback: if no proper splits, use all data for validation only
    if not val_files:
        val_files = [
            (str(r["video_path"]), label_to_id[r["label"]])
            for _, r in df.iterrows()
            if Path(str(r["video_path"])).exists()
        ]

    return class_names, train_files, val_files


def load_flame(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """Load FLAME dataset from manifest."""
    return load_from_manifest("flame", data_root)


def load_ufc_crime(data_root: str) -> Tuple[List[str], DataSplit, DataSplit]:
    """Load UCF-Crime dataset from manifest."""
    return load_from_manifest("ufc_crime", data_root)


_LOADERS = {
    "ucf101":       (load_ucf101,       "data/UCF101_data"),
    "hmdb51":       (load_hmdb51,       "data/HMDB51_data"),
    "diving48":     (load_diving48,     "data/Diving48_data"),
    "wlasl":        (load_wlasl,        "data/WLASL_data"),
    "wlasl100":     (load_wlasl100,     "data/WLASL_data"),
    "epic_kitchens":(load_epic_kitchens,"data/EPIC_data"),
    "autsl":        (load_autsl,        "data/AUTSL_data"),
    "driveact":     (load_driveact,     "data/DriveAct_data"),
    "kinetics400":  (load_kinetics400,  "data/Kinetics400_data"),
    "flame":        (load_flame,        "data/FLAME_data"),
    "ufc_crime":    (load_ufc_crime,    "data/UCFCrime_data"),
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
