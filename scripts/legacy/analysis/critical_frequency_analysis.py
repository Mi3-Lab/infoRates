#!/usr/bin/env python3
"""
Critical Frequency Analysis for Human Actions

Implements signal processing analysis to identify critical frequencies
for different human action types, following Nyquist-Shannon principles.
"""

import numpy as np
import pandas as pd
from scipy import signal, fft
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm

def analyze_action_frequencies(video_path, action_class):
    """
    Analyze temporal frequencies in a video action sequence.

    Returns:
    - dominant_frequencies: Main frequency components
    - nyquist_requirements: Minimum sampling rates needed
    - aliasing_risks: Potential aliasing at different frame rates
    """
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and compute motion features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    if len(frames) < 10:
        return None

    frames = np.array(frames)

    # Compute motion energy (temporal differences)
    motion_energy = np.abs(np.diff(frames.mean(axis=(1,2))))

    # Apply FFT to find frequency components
    freqs = fft.fftfreq(len(motion_energy))
    fft_vals = np.abs(fft.fft(motion_energy))

    # Find dominant frequencies (Hz, assuming 30fps source)
    fps = 30  # Assume 30fps source
    dominant_freqs = freqs[np.argsort(fft_vals)[-5:]] * fps
    dominant_freqs = dominant_freqs[dominant_freqs > 0]  # Remove negative

    # Calculate Nyquist requirements
    max_freq = np.max(dominant_freqs) if len(dominant_freqs) > 0 else 1.0
    nyquist_rate = 2 * max_freq

    return {
        'action_class': action_class,
        'dominant_frequencies': dominant_freqs.tolist(),
        'max_frequency': float(max_freq),
        'nyquist_rate': float(nyquist_rate),
        'recommended_fps': max(nyquist_rate, 5.0)  # Minimum 5fps
    }

def analyze_dataset_frequencies(dataset_path, dataset_name, sample_size=100):
    """
    Analyze critical frequencies across a dataset.
    """
    results = []

    # Get video files (implement based on dataset structure)
    if dataset_name == 'ucf101':
        video_pattern = "*.avi"
    elif dataset_name == 'kinetics400':
        video_pattern = "*.mp4"
    else:
        video_pattern = "*"

    video_files = list(Path(dataset_path).rglob(video_pattern))[:sample_size]

    print(f"Analyzing {len(video_files)} videos from {dataset_name}...")

    for video_path in tqdm(video_files):
        action_class = video_path.parent.name
        result = analyze_action_frequencies(str(video_path), action_class)
        if result:
            results.append(result)

    return results

def plot_frequency_analysis(results, output_dir):
    """
    Create visualizations of frequency analysis results.
    """
    df = pd.DataFrame(results)

    # Plot 1: Distribution of maximum frequencies by action type
    plt.figure(figsize=(12, 8))
    action_freqs = df.groupby('action_class')['max_frequency'].mean().sort_values(ascending=False)

    plt.bar(range(len(action_freqs)), action_freqs.values)
    plt.xticks(range(len(action_freqs)), action_freqs.index, rotation=45, ha='right')
    plt.xlabel('Action Class')
    plt.ylabel('Maximum Frequency (Hz)')
    plt.title('Critical Frequencies by Action Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'critical_frequencies_by_action.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Nyquist requirements distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['nyquist_rate'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Required Sampling Rate (fps)')
    plt.ylabel('Number of Actions')
    plt.title('Distribution of Nyquist Sampling Requirements')
    plt.axvline(df['nyquist_rate'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["nyquist_rate"].mean():.1f} fps')
    plt.legend()
    plt.savefig(output_dir / 'nyquist_requirements_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    df.to_csv(output_dir / 'frequency_analysis_results.csv', index=False)

    return df

def main():
    parser = argparse.ArgumentParser(description='Critical Frequency Analysis for Action Recognition')
    parser.add_argument('--dataset', required=True, choices=['ucf101', 'kinetics400', 'hmdb51'],
                       help='Dataset to analyze')
    parser.add_argument('--data-root', default='data', help='Root data directory')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of videos to sample')
    parser.add_argument('--output-dir', default='analysis/critical_frequencies',
                       help='Output directory')

    args = parser.parse_args()

    # Setup paths
    dataset_path = Path(args.data_root) / f"{args.dataset}_data"
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting critical frequency analysis for {args.dataset}")
    print(f"Dataset path: {dataset_path}")
    print(f"Sample size: {args.sample_size}")

    # Run analysis
    results = analyze_dataset_frequencies(dataset_path, args.dataset, args.sample_size)

    if results:
        # Generate plots and save results
        df_results = plot_frequency_analysis(results, output_dir)

        # Print summary
        print("
Analysis Summary:")
        print(f"Total actions analyzed: {len(results)}")
        print(".2f")
        print(".2f")
        print(".1f")

        print(f"\nResults saved to: {output_dir}")
    else:
        print("No valid results obtained. Check dataset path and video files.")

if __name__ == '__main__':
    main()