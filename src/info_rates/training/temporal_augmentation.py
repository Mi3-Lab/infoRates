"""
Temporal Robustness Augmentation (TRA) for Video Action Recognition.

This module implements training-time augmentation that randomly varies temporal
sampling parameters (coverage and stride) to improve model robustness against
temporal aliasing. Based on Nyquist-Shannon sampling theorem validation from
spectral analysis.

Key Features:
- Random coverage sampling: 25%, 50%, 75%, 100%
- Random stride sampling: 1, 2, 3 frames
- Compatible with TimeSformer, VideoMAE, ViViT models
- Seamless integration with existing training pipelines

References:
- Nyquist-Shannon Sampling Theorem
- Temporal aliasing mitigation via augmentation
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class TemporalRobustnessAugmentation:
    """
    Temporal Robustness Augmentation (TRA) strategy.
    
    Randomly samples temporal parameters during training to expose models
    to diverse temporal sampling conditions, improving robustness against
    aliasing at inference time.
    
    Args:
        coverage_range: List of coverage percentages to sample from (default: [25, 50, 75, 100])
        stride_range: List of stride values to sample from (default: [1, 2, 4, 8, 16])
        mode: Augmentation mode - "train" (random) or "val" (fixed)
        p_augment: Probability of applying augmentation (default: 0.5)
        
    Example:
        >>> tra = TemporalRobustnessAugmentation(mode="train")
        >>> coverage, stride = tra.sample()
        >>> print(f"Coverage: {coverage}%, Stride: {stride}")
        Coverage: 50%, Stride: 2
    """
    
    def __init__(
        self,
        coverage_range: List[int] = None,
        stride_range: List[int] = None,
        mode: str = "train",
        p_augment: float = 0.5,
    ):
        self.coverage_range = coverage_range or [25, 50, 75, 100]
        self.stride_range = stride_range or [1, 2, 4, 8, 16]
        self.mode = mode
        self.p_augment = p_augment
        
        # Validate inputs
        assert all(0 < c <= 100 for c in self.coverage_range), "Coverage must be in (0, 100]"
        assert all(s > 0 for s in self.stride_range), "Stride must be positive"
        assert mode in ["train", "val"], "Mode must be 'train' or 'val'"
        assert 0 <= p_augment <= 1, "p_augment must be in [0, 1]"
    
    def sample(self) -> Tuple[int, int]:
        """
        Sample temporal parameters.
        
        Returns:
            Tuple of (coverage_percent, stride)
        """
        if self.mode == "val":
            # Validation: use full coverage and minimum stride (no augmentation)
            return 100, 1
        
        # Training: randomly augment with probability p_augment
        if np.random.rand() < self.p_augment:
            coverage = np.random.choice(self.coverage_range)
            stride = np.random.choice(self.stride_range)
        else:
            # No augmentation
            coverage = 100
            stride = 1
            
        return coverage, stride
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply temporal augmentation to frame sequence.
        
        Args:
            frames: Input frames with shape (T, H, W, C)
            
        Returns:
            Augmented frames with potentially reduced temporal extent
        """
        coverage, stride = self.sample()
        return self._apply_temporal_sampling(frames, coverage, stride)
    
    @staticmethod
    def _apply_temporal_sampling(
        frames: np.ndarray,
        coverage: int,
        stride: int
    ) -> np.ndarray:
        """
        Apply temporal subsampling to frame sequence.
        
        Args:
            frames: Input frames (T, H, W, C)
            coverage: Percentage of clip to use [1-100]
            stride: Temporal stride [1, 2, 3, ...]
            
        Returns:
            Subsampled frames
        """
        n_total = len(frames)
        n_keep = max(1, int(n_total * coverage / 100))
        return frames[:n_keep:stride]


class TRADataset(Dataset):
    """
    PyTorch Dataset wrapper with Temporal Robustness Augmentation.
    
    Wraps an existing video dataset and applies TRA during training.
    Compatible with standard video datasets that return (frames, label) pairs.
    
    Args:
        video_paths: List of video file paths
        labels: List of corresponding labels (class indices)
        processor: Hugging Face image processor for frame preprocessing
        num_frames: Target number of frames to extract
        size: Spatial size for resizing (height and width)
        tra: TemporalRobustnessAugmentation instance
        class_names: Optional list of class names for label mapping
        
    Example:
        >>> tra = TemporalRobustnessAugmentation(mode="train")
        >>> dataset = TRADataset(
        ...     video_paths=train_paths,
        ...     labels=train_labels,
        ...     processor=processor,
        ...     tra=tra
        ... )
        >>> frames, label = dataset[0]
    """
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        processor: Callable,
        num_frames: int = 8,
        size: int = 224,
        tra: Optional[TemporalRobustnessAugmentation] = None,
        class_names: Optional[List[str]] = None,
    ):
        assert len(video_paths) == len(labels), "Mismatch between paths and labels"
        
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.num_frames = num_frames
        self.size = size
        self.tra = tra or TemporalRobustnessAugmentation(mode="val")  # Default: no augmentation
        self.class_names = class_names or []
        
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a video, apply TRA, and preprocess frames.
        
        Returns:
            Dictionary with keys:
                - pixel_values: Preprocessed frame tensor (C, T, H, W)
                - labels: Class label tensor (scalar)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load video and extract base frames
            frames = self._load_video(video_path)
            
            # Apply temporal augmentation
            frames = self.tra(frames)
            
            # Ensure we have enough frames
            frames = self._ensure_num_frames(frames)
            
            # Resize frames
            frames = [cv2.resize(f, (self.size, self.size)) for f in frames]
            
            # Preprocess with model processor
            inputs = self.processor(frames, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["labels"] = torch.tensor(label, dtype=torch.long)
            
            return inputs
            
        except Exception as e:
            # Fallback: return zero tensor to avoid crashing during training
            print(f"Warning: Failed to load video {video_path}: {e}")
            dummy_frames = np.zeros((self.num_frames, self.size, self.size, 3), dtype=np.uint8)
            inputs = self.processor(list(dummy_frames), return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["labels"] = torch.tensor(label, dtype=torch.long)
            return inputs
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """
        Load video and extract uniformly sampled frames.
        
        Extracts 4x the target number of frames to provide sufficient
        temporal resolution for augmentation.
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Sample 4x target frames to allow for aggressive subsampling
        base_n = min(total_frames, self.num_frames * 4)
        indices = np.linspace(0, max(total_frames - 1, 0), base_n).astype(int)
        
        frames = vr.get_batch(indices).asnumpy()
        return frames
    
    def _ensure_num_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Ensure we have exactly num_frames by resampling or padding.
        """
        current_n = len(frames)
        
        if current_n >= self.num_frames:
            # Resample to target number
            indices = np.linspace(0, current_n - 1, self.num_frames).astype(int)
            return frames[indices]
        else:
            # Repeat frames if we have too few
            repeats = (self.num_frames // current_n) + 1
            frames = np.repeat(frames, repeats, axis=0)
            return frames[:self.num_frames]


class TRACollator:
    """
    Custom collate function for batching TRADataset samples.
    
    Handles variable-length sequences that may result from temporal augmentation.
    """
    
    def __init__(self, processor: Callable):
        self.processor = processor
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples into model inputs.
        
        Args:
            batch: List of dictionaries from TRADataset
            
        Returns:
            Batched dictionary ready for model forward pass
        """
        # Stack all tensors
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# Utility functions for integration with existing pipelines

def create_tra_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    processor: Callable,
    batch_size: int = 8,
    num_frames: int = 8,
    size: int = 224,
    coverage_range: List[int] = None,
    stride_range: List[int] = None,
    p_augment: float = 0.5,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation DataLoaders with TRA.
    
    Args:
        train_paths: Training video paths
        train_labels: Training labels
        val_paths: Validation video paths
        val_labels: Validation labels
        processor: Hugging Face image processor
        batch_size: Batch size
        num_frames: Number of frames per video
        size: Spatial resize dimension
        coverage_range: Coverage values for TRA
        stride_range: Stride values for TRA
        p_augment: Augmentation probability
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create augmentation strategies
    tra_train = TemporalRobustnessAugmentation(
        coverage_range=coverage_range,
        stride_range=stride_range,
        mode="train",
        p_augment=p_augment,
    )
    
    tra_val = TemporalRobustnessAugmentation(
        mode="val",  # No augmentation during validation
    )
    
    # Create datasets
    train_dataset = TRADataset(
        video_paths=train_paths,
        labels=train_labels,
        processor=processor,
        num_frames=num_frames,
        size=size,
        tra=tra_train,
    )
    
    val_dataset = TRADataset(
        video_paths=val_paths,
        labels=val_labels,
        processor=processor,
        num_frames=num_frames,
        size=size,
        tra=tra_val,
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


def get_tra_stats(tra: TemporalRobustnessAugmentation, n_samples: int = 1000) -> Dict[str, float]:
    """
    Calculate expected statistics of TRA sampling distribution.
    
    Useful for understanding augmentation behavior and reporting in papers.
    
    Args:
        tra: TemporalRobustnessAugmentation instance
        n_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        Dictionary with statistics:
            - mean_coverage: Average coverage percentage
            - std_coverage: Standard deviation of coverage
            - mean_stride: Average stride
            - std_stride: Standard deviation of stride
            - augmentation_rate: Fraction of samples that are augmented
    """
    coverages = []
    strides = []
    
    for _ in range(n_samples):
        cov, stride = tra.sample()
        coverages.append(cov)
        strides.append(stride)
    
    coverages = np.array(coverages)
    strides = np.array(strides)
    
    # Calculate augmentation rate (how often we deviate from 100% coverage, stride 1)
    is_augmented = (coverages != 100) | (strides != 1)
    augmentation_rate = is_augmented.mean()
    
    return {
        "mean_coverage": coverages.mean(),
        "std_coverage": coverages.std(),
        "mean_stride": strides.mean(),
        "std_stride": strides.std(),
        "augmentation_rate": augmentation_rate,
    }
