import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Any
from pathlib import Path


def list_classes(labels_path: str) -> List[str]:
    """Load class names from Something-Something V2 labels.json."""
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    if isinstance(labels_data, list):
        return labels_data
    else:
        # Sort by numeric value to ensure consistent ordering
        return [k for k, v in sorted(labels_data.items(), key=lambda x: int(x[1]))]


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load manifest CSV file."""
    return pd.read_csv(manifest_path)


def get_train_val_test_manifests(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test manifests for Something-Something V2."""
    base_path = Path(data_root) / "labels"
    
    # Load JSON files
    with open(base_path / "train.json", 'r') as f:
        train_data = json.load(f)
    with open(base_path / "validation.json", 'r') as f:
        val_data = json.load(f)
    with open(base_path / "test.json", 'r') as f:
        test_data = json.load(f)
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    # Ensure 'label' column contains the template (unique) to avoid duplicate column names
    for df in (train_df, val_df, test_df):
        if 'template' in df.columns:
            # Use the human-readable template as the canonical label
            df['label'] = df['template']
            # Drop template column in-place to avoid duplicate 'label' columns
            df.drop(columns=['template'], inplace=True)
        # Rename id to full video path later
    
    # Rename id column to match expected format (video path will be expanded below)
    train_df = train_df.rename(columns={'id': 'video_path'})
    val_df = val_df.rename(columns={'id': 'video_path'})
    test_df = test_df.rename(columns={'id': 'video_path'})
    
    # Convert video paths to full paths
    video_root = Path(data_root) / "videos"
    train_df['video_path'] = train_df['video_path'].apply(lambda x: str(video_root / f"{x}.webm"))
    val_df['video_path'] = val_df['video_path'].apply(lambda x: str(video_root / f"{x}.webm"))
    test_df['video_path'] = test_df['video_path'].apply(lambda x: str(video_root / f"{x}.webm"))
    
    return train_df, val_df, test_df


def get_class_mapping(labels_path: str) -> Dict[str, int]:
    """Get mapping from class name to numeric label."""
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    if isinstance(labels_data, list):
        return {name: idx for idx, name in enumerate(labels_data)}
    else:
        return {k: int(v) for k, v in labels_data.items()}


def get_numeric_labels(manifest_df: pd.DataFrame, class_mapping: Dict[str, int]) -> pd.DataFrame:
    """Convert template labels to numeric labels, handling template formatting differences.

    The Something-Something manifests use templates like "Holding [something] next to [something]",
    while the labels.json keys may be formatted without square brackets (e.g., "Holding something next to something").
    This function normalizes both sides before mapping.
    """
    df = manifest_df.copy()
    # Ensure label is string
    df['label'] = df['label'].astype(str)

    # Normalize label text by removing square brackets and trimming whitespace
    df['label_norm'] = df['label'].str.replace(r"[\[\]]", "", regex=True).str.strip()

    # Normalize class mapping keys similarly
    normalized_map = { (k.replace('[','').replace(']','').strip()): v for k, v in class_mapping.items() }

    # Map normalized labels to numeric ids
    df['label'] = df['label_norm'].map(normalized_map)
    df.drop(columns=['label_norm'], inplace=True)
    return df