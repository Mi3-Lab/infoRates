#!/usr/bin/env python3
"""
End-to-End Spectral Analysis Pipeline
=====================================

Comprehensive workflow:
  1. Load per-class sensitivity from evaluation results
  2. Extract optical flow from sample videos
  3. Compute temporal FFT and spectral metrics
  4. Correlate spectral properties with empirical aliasing sensitivity
  5. Generate publication-quality validation plots

This bridges theory (Nyquist-Shannon) with empirical findings.

Usage:
    # Using existing evaluation results + sample videos
    python scripts/run_spectral_analysis.py \
      --sensitivity-csv evaluations/ucf101/vivit/vivit_per_class.csv \
      --dataset-root data/UCF101_data/UCF-101 \
      --output-dir evaluations/spectral_validation

    # Quick demo (synthetic data for immediate visualization)
    python scripts/run_spectral_analysis.py --output-dir evaluations/spectral_demo
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main(args):
    # Add src to path
    project_root = Path(__file__).parent.parent
    src_root = project_root / "src"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_root))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🔬 SPECTRAL-ALIASING VALIDATION PIPELINE")
    print("="*80)
    
    # Import here so dependencies are available
    from info_rates.analysis.spectral_analysis import (
        SpectralAnalyzer,
        OpticalFlowExtractor,
        TemporalFFT,
        aggregate_spectral_metrics,
    )
    
    print("\n✓ Spectral analysis modules imported successfully")
    
    # Call the correlation script
    print("\nRunning spectral correlation analysis...")
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "analysis" / "spectral_correlation.py"),
    ]
    
    # Pass through relevant arguments
    if args.sensitivity_csv:
        cmd.extend(["--sensitivity-csv", args.sensitivity_csv])
    if args.dataset_root:
        cmd.extend(["--dataset-root", args.dataset_root])
    
    cmd.extend(["--output-dir", args.output_dir])
    
    if args.optical_flow_method:
        cmd.extend(["--optical-flow-method", args.optical_flow_method])
    if args.fft_method:
        cmd.extend(["--fft-method", args.fft_method])
    if args.max_videos_per_class:
        cmd.extend(["--max-videos-per-class", str(args.max_videos_per_class)])
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ SPECTRAL ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to: {output_dir}/")
        print("\nKey files:")
        print("  - correlation_analysis.json: Correlation coefficients & p-values")
        print("  - spectral_validation_summary.csv: Per-class spectral summary")
        print("  - 01_spectral_profiles.png: Spectral characteristics by class")
        print("  - 02_correlation_scatter.png: Frequency vs. sensitivity scatter plots")
        print("  - 03_sensitivity_tiers_spectral.png: Ranked sensitivity with spectral overlay")
        return 0
    else:
        print("\n⚠️ Spectral analysis failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end spectral validation of Nyquist-Shannon principle"
    )
    parser.add_argument(
        "--sensitivity-csv", type=str, default=None,
        help="Path to per-class sensitivity CSV from evaluation"
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Root directory of video dataset for spectral analysis"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluations/spectral_analysis",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--optical-flow-method", type=str, default="farneback",
        choices=["farneback", "lk"],
        help="Optical flow method to use"
    )
    parser.add_argument(
        "--fft-method", type=str, default="welch",
        choices=["fft", "welch"],
        help="FFT estimation method"
    )
    parser.add_argument(
        "--max-videos-per-class", type=int, default=10,
        help="Maximum videos per class to analyze"
    )
    
    args = parser.parse_args()
    sys.exit(main(args))
