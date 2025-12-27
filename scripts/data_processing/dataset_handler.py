#!/usr/bin/env python3
"""
Dataset Handler for Multi-Dataset Support

Provides unified interface for loading different video datasets:
- UCF-101
- Kinetics400
- HMDB51
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

class DatasetHandler:
    """Handles dataset-specific loading and manifest creation."""

    DATASETS = {
        "ucf101": {
            "name": "UCF-101",
            "data_dir": "data/UCF101_data",
            "manifest_dir": "data/UCF101_data/manifests",
            "results_dir": "data/UCF101_data/results",
            "video_root": "data/UCF101_data/UCF-101",
            "classes": 101,
            "wandb_project": "inforates-ucf101"
        },
        "kinetics400": {
            "name": "Kinetics400",
            "data_dir": "data/Kinetics400_data",
            "manifest_dir": "data/Kinetics400_data/manifests",
            "results_dir": "data/Kinetics400_data/results",
            "video_root": "data/Kinetics400_data/k4testset/videos_val",
            "val_list": "data/Kinetics400_data/k4testset/kinetics400_val_list_videos.txt",
            "classes": 400,
            "wandb_project": "inforates-kinetics400"
        },
        "hmdb51": {
            "name": "HMDB51",
            "data_dir": "data/HMDB51_data",
            "manifest_dir": "data/HMDB51_data/manifests",
            "results_dir": "data/HMDB51_data/results",
            "video_root": "data/HMDB51_data/videos",
            "classes": 51,
            "wandb_project": "inforates-hmdb51"
        }
    }

    @staticmethod
    def get_dataset_config(dataset_name: str) -> Dict:
        """Get configuration for a specific dataset."""
        if dataset_name not in DatasetHandler.DATASETS:
            available = list(DatasetHandler.DATASETS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
        return DatasetHandler.DATASETS[dataset_name]

    @staticmethod
    def get_default_manifest(dataset_name: str) -> str:
        """Get default manifest path for a dataset."""
        config = DatasetHandler.get_dataset_config(dataset_name)
        manifest_dir = config["manifest_dir"]

        if dataset_name == "ucf101":
            return f"{manifest_dir}/ucf101_50f.csv"
        elif dataset_name == "kinetics400":
            return f"{manifest_dir}/kinetics400_val.csv"
        elif dataset_name == "hmdb51":
            return f"{manifest_dir}/hmdb51_test.csv"
        else:
            return f"{manifest_dir}/{dataset_name}_manifest.csv"

    @staticmethod
    def build_kinetics400_manifest(val_list_path: str, video_dir: str, output_path: str) -> pd.DataFrame:
        """Build manifest for Kinetics400 validation set."""
        print(f"Building Kinetics400 manifest from {val_list_path}")

        videos = []
        labels = []

        with open(val_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_file = parts[0]
                    label = int(parts[1])
                    video_path = os.path.join(video_dir, video_file)

                    if os.path.exists(video_path):
                        videos.append(video_path)
                        # Convert integer label to string for compatibility with evaluation code
                        labels.append(str(label))

        df = pd.DataFrame({
            'video_path': videos,
            'label': labels
        })

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved manifest with {len(df)} videos to {output_path}")

        return df

    @staticmethod
    def load_or_build_manifest(dataset_name: str) -> Tuple[pd.DataFrame, str]:
        """Load existing manifest or build new one for the dataset."""
        config = DatasetHandler.get_dataset_config(dataset_name)
        manifest_path = DatasetHandler.get_default_manifest(dataset_name)

        if os.path.exists(manifest_path):
            print(f"Loading existing manifest: {manifest_path}")
            df = pd.read_csv(manifest_path)
        else:
            print(f"Building new manifest for {dataset_name}")

            if dataset_name == "kinetics400":
                val_list = config["val_list"]
                video_dir = config["video_root"]
                if os.path.exists(val_list):
                    df = DatasetHandler.build_kinetics400_manifest(val_list, video_dir, manifest_path)
                else:
                    raise FileNotFoundError(f"Kinetics400 validation list not found: {val_list}")
            else:
                raise NotImplementedError(f"Auto-building manifest not implemented for {dataset_name}")

        return df, manifest_path

    @staticmethod
    def get_model_defaults(dataset_name: str) -> Dict:
        """Get model-specific defaults for a dataset."""
        config = DatasetHandler.get_dataset_config(dataset_name)

        # Model ID mappings for different datasets
        model_ids = {
            "ucf101": {
                "timesformer": "facebook/timesformer-base-finetuned-k400",  # Pre-trained on K400, can use for UCF
                "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
                "vivit": "google/vivit-b-16x2"
            },
            "kinetics400": {
                "timesformer": "facebook/timesformer-base-finetuned-k400",
                "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
                "vivit": "google/vivit-b-16x2"
            },
            "hmdb51": {
                "timesformer": "facebook/timesformer-base-finetuned-k400",
                "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
                "vivit": "google/vivit-b-16x2"
            }
        }

        return {
            "model_ids": model_ids.get(dataset_name, {}),
            "num_classes": config["classes"],
            "wandb_project": config["wandb_project"],
            "results_dir": config["results_dir"]
        }