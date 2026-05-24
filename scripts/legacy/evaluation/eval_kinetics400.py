#!/usr/bin/env python3
"""
Simple evaluation script for TimeSformer on Kinetics400 validation set.

This script evaluates the pre-trained TimeSformer model on Kinetics400
without any fine-tuning, to establish baseline performance.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from decord import VideoReader, cpu
import cv2
import random

from PIL import Image

def load_video_frames(video_path, num_frames=8, frame_size=224):
    """Load and preprocess video frames for TimeSformer using decord."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames < num_frames:
            print(f"Warning: Video {video_path} has only {total_frames} frames, needed {num_frames}")
            return None

        # Sample frames uniformly
        idx = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)
        frames = vr.get_batch(idx).asnumpy()

        # Convert to list of PIL images
        pil_frames = []
        for frame in frames:
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame)
            pil_frames.append(pil_frame)

        return pil_frames
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

def evaluate_kinetics400(model, processor, val_list_path, video_dir, num_samples=100, device="cuda"):
    """Evaluate TimeSformer on Kinetics400 validation set."""

    # Read validation list
    with open(val_list_path, 'r') as f:
        val_list = [line.strip().split() for line in f.readlines()]

    # Sample a subset for quick evaluation
    if num_samples and num_samples < len(val_list):
        val_list = random.sample(val_list, num_samples)
        print(f"Evaluating on {num_samples} randomly sampled videos")
    else:
        print(f"Evaluating on all {len(val_list)} videos")

    results = []
    correct = 0
    total = 0

    model.eval()
    model.to(device)

    for video_file, label_str in tqdm(val_list, desc="Evaluating"):
        video_path = os.path.join(video_dir, video_file)
        true_label = int(label_str)

        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found, skipping")
            continue

        # Load and preprocess video
        frames = load_video_frames(video_path, num_frames=8)
        if frames is None:
            continue

        # Prepare inputs - frames is already a list of PIL images
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = outputs.logits.argmax(-1).item()

        # Record result
        is_correct = predicted_label == true_label
        results.append({
            'video': video_file,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': is_correct
        })

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(".4f")

    return results, accuracy

def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Paths
    base_dir = "/home/wesleyferreiramaia/data/infoRates/data/Kinetics400_data/k4testset"
    val_list_path = os.path.join(base_dir, "kinetics400_val_list_videos.txt")
    video_dir = os.path.join(base_dir, "videos_val")

    # Model
    model_name = "facebook/timesformer-base-finetuned-k400"
    print(f"Loading {model_name}...")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForVideoClassification.from_pretrained(model_name)

    # Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Quick evaluation on 100 samples first
    results, accuracy = evaluate_kinetics400(
        model, processor, val_list_path, video_dir,
        num_samples=100, device=device
    )

    # Save results
    results_dir = "/home/wesleyferreiramaia/data/infoRates/data/Kinetics400_data/results"
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "timesformer_kinetics400_quick_eval.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    print("\nQuick evaluation complete!")
    print(f"Sample accuracy: {accuracy:.4f}")
    print("Run with num_samples=None in evaluate_kinetics400() for full evaluation")

if __name__ == "__main__":
    main()