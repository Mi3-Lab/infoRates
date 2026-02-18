"""
Spectral Analysis of Action Classes
===================================

Bridges theory (Nyquist-Shannon) with empirical results by computing:
  1. Optical flow from videos
  2. Temporal FFT for each spatial region
  3. Dominant frequencies & spectral energy distribution
  4. Correlation with aliasing sensitivity (empirically observed)

This validates the hypothesis:
  - High-frequency actions → High dominant frequencies → Require dense sampling
  - Low-frequency actions → Low dominant frequencies → Tolerate subsampling

Theory: A signal with maximum frequency f_max requires sampling rate ≥ 2*f_max (Nyquist-Shannon)
Applied to video: Action dynamics (motion energy over time) have specific frequency content
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from decord import VideoReader, cpu
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SpectralMetrics:
    """Container for spectral analysis results for a single video."""
    video_path: str
    label: str
    n_frames: int
    fps: float
    dominant_frequency: float  # Hz
    dominant_frequency_bin: int
    peak_power: float
    spectral_energy_ratio: float  # Power in [1-5Hz] / total power
    spectral_centroid: float  # Weighted mean frequency
    spectral_flatness: float  # Wiener entropy (0=tonal, 1=flat)
    power_spectrum: np.ndarray  # Full FFT magnitude
    frequencies: np.ndarray  # Frequency bins


class OpticalFlowExtractor:
    """Extract optical flow magnitude from video."""
    
    @staticmethod
    def extract_from_frames(frames: np.ndarray, method: str = "farneback") -> np.ndarray:
        """
        Compute optical flow magnitude between consecutive frames.
        
        Args:
            frames: (T, H, W, 3) array of video frames
            method: "farneback" (dense) or "lk" (Lucas-Kanade sparse)
            
        Returns:
            flow_magnitude: (T-1, H, W) array of optical flow magnitudes
        """
        if len(frames) < 2:
            return np.array([])
        
        # Convert to grayscale
        gray_frames = np.array([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY) 
                                for f in frames])
        
        flow_magnitudes = []
        
        if method == "farneback":
            for i in range(len(gray_frames) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[i], 
                    gray_frames[i+1],
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    n8=False,
                    poly_n=5,
                    poly_sigma=1.1
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitudes.append(magnitude)
        
        elif method == "lk":
            # Sparse Lucas-Kanade
            for i in range(len(gray_frames) - 1):
                p0 = cv2.goodFeaturesToTrack(
                    gray_frames[i], 
                    maxCorners=100, 
                    qualityLevel=0.01, 
                    minDistance=10
                )
                if p0 is None:
                    flow_magnitudes.append(np.zeros_like(gray_frames[i], dtype=np.float32))
                    continue
                    
                p1, status, err = cv2.calcOpticalFlowLK(
                    gray_frames[i], 
                    gray_frames[i+1], 
                    p0, 
                    winSize=(15, 15)
                )
                
                # Reconstruct dense flow estimate
                magnitude = np.zeros_like(gray_frames[i], dtype=np.float32)
                if status is not None:
                    valid = status.flatten() == 1
                    magnitude[p0[valid, 0, 1].astype(int), 
                             p0[valid, 0, 0].astype(int)] = np.linalg.norm(
                        p1[valid] - p0[valid], axis=2
                    ).flatten()
                flow_magnitudes.append(magnitude)
        
        return np.array(flow_magnitudes)
    
    @staticmethod
    def extract_from_video(video_path: str, subsample: int = 1, 
                          method: str = "farneback") -> np.ndarray:
        """
        Extract optical flow directly from video file.
        
        Args:
            video_path: Path to video file
            subsample: Sample every Nth frame (for speed)
            method: "farneback" or "lk"
            
        Returns:
            flow_magnitude: (T-1, H, W) array
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            indices = np.arange(0, len(vr), subsample)
            frames = vr.get_batch(indices).asnumpy()
            return OpticalFlowExtractor.extract_from_frames(frames, method=method)
        except Exception as e:
            print(f"⚠️ Error extracting optical flow from {video_path}: {e}")
            return np.array([])


class TemporalFFT:
    """Compute temporal FFT of optical flow."""
    
    @staticmethod
    def compute_single_channel_fft(time_series: np.ndarray, fps: float = 30.0,
                                   method: str = "welch") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of a 1D time series.
        
        Args:
            time_series: (T,) array of temporal values
            fps: Frames per second (for frequency scaling)
            method: "fft" (raw FFT) or "welch" (Welch's method, smoother)
            
        Returns:
            frequencies: Frequency bins (Hz)
            power_spectrum: Power/magnitude values
        """
        if len(time_series) < 2:
            return np.array([]), np.array([])
        
        if method == "fft":
            fft_vals = fft(time_series - np.mean(time_series))
            power = np.abs(fft_vals) ** 2
            freqs = fftfreq(len(time_series), 1 / fps)
            # Keep only positive frequencies
            pos_mask = freqs >= 0
            return freqs[pos_mask], power[pos_mask]
        
        elif method == "welch":
            # Welch's method provides smoother spectral estimate
            segment_length = min(len(time_series) // 2, 256)
            freqs, power = welch(time_series, fps=fps, nperseg=segment_length)
            return freqs, power
    
    @staticmethod
    def compute_optical_flow_spectrum(flow_magnitude: np.ndarray, 
                                     fps: float = 30.0,
                                     method: str = "welch") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute temporal FFT of optical flow magnitude.
        
        Aggregates spatial dimensions: computes mean flow magnitude over space,
        then FFT over time.
        
        Args:
            flow_magnitude: (T, H, W) optical flow magnitude
            fps: Frames per second
            method: "fft" or "welch"
            
        Returns:
            frequencies: (F,) frequency bins
            power_spectrum: (F,) power values
        """
        if len(flow_magnitude) == 0:
            return np.array([]), np.array([])
        
        # Temporal average of spatial flow (collapse spatial dims)
        temporal_signal = flow_magnitude.mean(axis=(1, 2))
        
        return TemporalFFT.compute_single_channel_fft(temporal_signal, fps=fps, 
                                                     method=method)


class SpectralAnalyzer:
    """Main analyzer: FFT + spectral characterization."""
    
    @staticmethod
    def analyze_video(video_path: str, label: str, 
                     optical_flow_method: str = "farneback",
                     fft_method: str = "welch",
                     fps: float = 30.0,
                     subsample: int = 2) -> Optional[SpectralMetrics]:
        """
        Full pipeline: optical flow → FFT → spectral metrics.
        
        Args:
            video_path: Path to video
            label: Action class label
            optical_flow_method: "farneback" or "lk"
            fft_method: "fft" or "welch"
            fps: Sampling rate (frames per second)
            subsample: Sample every Nth frame for speed
            
        Returns:
            SpectralMetrics object with full spectral characterization
        """
        try:
            # Step 1: Extract optical flow
            flow_mag = OpticalFlowExtractor.extract_from_video(
                video_path, 
                subsample=subsample,
                method=optical_flow_method
            )
            
            if len(flow_mag) == 0:
                return None
            
            # Step 2: Compute FFT
            freqs, power = TemporalFFT.compute_optical_flow_spectrum(
                flow_mag, 
                fps=fps / subsample,  # Adjust for subsampling
                method=fft_method
            )
            
            if len(freqs) == 0:
                return None
            
            # Step 3: Extract spectral metrics
            metrics = SpectralAnalyzer._extract_metrics(
                freqs, power, video_path, label, len(flow_mag), fps
            )
            
            return metrics
        
        except Exception as e:
            print(f"⚠️ Error analyzing {video_path}: {e}")
            return None
    
    @staticmethod
    def _extract_metrics(frequencies: np.ndarray, power_spectrum: np.ndarray,
                        video_path: str, label: str,
                        n_frames: int, fps: float) -> SpectralMetrics:
        """Extract statistically meaningful metrics from power spectrum."""
        
        # Dominant frequency (peak)
        dominant_bin = np.argmax(power_spectrum)
        dominant_freq = frequencies[dominant_bin]
        peak_power = power_spectrum[dominant_bin]
        
        # Spectral energy ratio: How much power is in "low-frequency" band [1-5Hz]
        # (typical for human actions)
        low_freq_mask = (frequencies >= 1.0) & (frequencies <= 5.0)
        energy_low = power_spectrum[low_freq_mask].sum() if low_freq_mask.sum() > 0 else 0
        energy_total = power_spectrum.sum()
        spectral_energy_ratio = energy_low / (energy_total + 1e-10)
        
        # Spectral centroid (weighted mean frequency)
        spectral_centroid = np.sum(frequencies * power_spectrum) / (np.sum(power_spectrum) + 1e-10)
        
        # Spectral flatness (Wiener entropy)
        # High value = flat (noise-like), Low value = tonal (periodic)
        p_normalized = power_spectrum / (power_spectrum.sum() + 1e-10)
        spectral_flatness = -np.sum(p_normalized * np.log(p_normalized + 1e-10)) / np.log(len(power_spectrum))
        
        return SpectralMetrics(
            video_path=str(video_path),
            label=label,
            n_frames=n_frames,
            fps=fps,
            dominant_frequency=float(dominant_freq),
            dominant_frequency_bin=int(dominant_bin),
            peak_power=float(peak_power),
            spectral_energy_ratio=float(spectral_energy_ratio),
            spectral_centroid=float(spectral_centroid),
            spectral_flatness=float(spectral_flatness),
            power_spectrum=power_spectrum,
            frequencies=frequencies
        )
    
    @staticmethod
    def batch_analyze_videos(video_paths: List[str], labels: List[str],
                            max_videos_per_class: int = 10,
                            **kwargs) -> Dict[str, List[SpectralMetrics]]:
        """
        Analyze multiple videos, grouped by class.
        
        Args:
            video_paths: List of video file paths
            labels: Corresponding action class labels
            max_videos_per_class: Limit samples per class for speed
            **kwargs: Passed to analyze_video()
            
        Returns:
            {class_name: [SpectralMetrics, ...], ...}
        """
        results = {}
        class_counter = {}
        
        for vp, label in zip(video_paths, labels):
            # Limit per-class
            class_counter[label] = class_counter.get(label, 0) + 1
            if class_counter[label] > max_videos_per_class:
                continue
            
            print(f"Analyzing {label} / {Path(vp).name}...")
            metrics = SpectralAnalyzer.analyze_video(vp, label, **kwargs)
            
            if metrics:
                if label not in results:
                    results[label] = []
                results[label].append(metrics)
        
        return results


def aggregate_spectral_metrics(metrics_by_class: Dict[str, List[SpectralMetrics]]
                               ) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-video metrics to per-class summary statistics.
    
    Returns:
        {class_name: {
            'mean_dominant_freq': float,
            'std_dominant_freq': float,
            'mean_spectral_centroid': float,
            'mean_energy_ratio': float,
            'mean_flatness': float,
            'n_videos': int,
            ...
        }, ...}
    """
    class_summaries = {}
    
    for class_name, metrics_list in metrics_by_class.items():
        if not metrics_list:
            continue
        
        dominant_freqs = np.array([m.dominant_frequency for m in metrics_list])
        centroids = np.array([m.spectral_centroid for m in metrics_list])
        energy_ratios = np.array([m.spectral_energy_ratio for m in metrics_list])
        flatness = np.array([m.spectral_flatness for m in metrics_list])
        
        class_summaries[class_name] = {
            'n_videos': len(metrics_list),
            'mean_dominant_freq': float(dominant_freqs.mean()),
            'std_dominant_freq': float(dominant_freqs.std()),
            'median_dominant_freq': float(np.median(dominant_freqs)),
            'q25_dominant_freq': float(np.percentile(dominant_freqs, 25)),
            'q75_dominant_freq': float(np.percentile(dominant_freqs, 75)),
            
            'mean_spectral_centroid': float(centroids.mean()),
            'std_spectral_centroid': float(centroids.std()),
            
            'mean_energy_ratio': float(energy_ratios.mean()),
            'mean_flatness': float(flatness.mean()),
        }
    
    return class_summaries


if __name__ == "__main__":
    # Example: Analyze a few videos
    print("✅ Spectral analysis module loaded")
    print("Usage: import and call SpectralAnalyzer.analyze_video() or batch_analyze_videos()")
