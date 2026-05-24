#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Sweep for Temporal Parameters

Systematically tests frame rates and clip durations across datasets and models
as specified in research milestones.
"""

import os
import sys
import argparse
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import yaml

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_single_evaluation(model_name, dataset_name, frame_rate, clip_duration, coverage, stride):
    """
    Run single evaluation configuration.
    """
    # This would call the existing evaluation script with specific parameters
    cmd = f"""
    python scripts/evaluation/run_eval_multimodel.py \
        --model {model_name} \
        --dataset {dataset_name} \
        --frame-rate {frame_rate} \
        --clip-duration {clip_duration} \
        --coverage {coverage} \
        --stride {stride} \
        --output-dir evaluations/hyperparameter_sweep/{model_name}_{dataset_name}
    """

    print(f"Running: {model_name} on {dataset_name} (fps={frame_rate}, duration={clip_duration}s)")
    result = os.system(cmd)

    return {
        'model': model_name,
        'dataset': dataset_name,
        'frame_rate': frame_rate,
        'clip_duration': clip_duration,
        'coverage': coverage,
        'stride': stride,
        'success': result == 0
    }

def hyperparameter_sweep(models, datasets, frame_rates, clip_durations, coverages, strides, max_workers=4):
    """
    Run comprehensive hyperparameter sweep.
    """
    # Create all combinations
    combinations = list(itertools.product(
        models, datasets, frame_rates, clip_durations, coverages, strides
    ))

    print(f"Total configurations to test: {len(combinations)}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Frame rates: {frame_rates}")
    print(f"Clip durations: {clip_durations}")
    print(f"Coverages: {coverages}")
    print(f"Strides: {strides}")

    results = []

    # Run evaluations (parallel if possible)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_evaluation, *combo)
            for combo in combinations
        ]

        for future in futures:
            result = future.result()
            results.append(result)
            print(f"Completed: {result}")

    return results

def generate_sweep_report(results, output_dir):
    """
    Generate comprehensive report with graphs and tables.
    """
    df = pd.DataFrame(results)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    df.to_csv(output_dir / "hyperparameter_sweep_results.csv", index=False)

    # Generate summary statistics
    success_rate = df['success'].mean() * 100
    print(".1f")

    # Group by different parameters and compute statistics
    param_groups = ['model', 'dataset', 'frame_rate', 'clip_duration']

    for param in param_groups:
        if param in df.columns:
            success_by_param = df.groupby(param)['success'].agg(['mean', 'count'])
            success_by_param['success_rate'] = success_by_param['mean'] * 100
            success_by_param.to_csv(output_dir / f"success_by_{param}.csv")

            print(f"\nSuccess rates by {param}:")
            print(success_by_param[['success_rate', 'count']])

    # Generate performance analysis (if accuracy data available)
    # This would require parsing the actual evaluation results

    print(f"\nReport generated in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Hyperparameter Sweep')
    parser.add_argument('--models', nargs='+', default=['timesformer', 'videomae', 'vivit'],
                       help='Models to test')
    parser.add_argument('--datasets', nargs='+', default=['ucf101', 'kinetics400'],
                       help='Datasets to test')
    parser.add_argument('--frame-rates', nargs='+', type=float,
                       default=[5, 10, 15, 30], help='Frame rates to test (fps)')
    parser.add_argument('--clip-durations', nargs='+', type=float,
                       default=[1, 2, 4, 8], help='Clip durations to test (seconds)')
    parser.add_argument('--coverages', nargs='+', type=int,
                       default=[25, 50, 75, 100], help='Coverage percentages')
    parser.add_argument('--strides', nargs='+', type=int,
                       default=[1, 2, 4, 8], help='Stride values')
    parser.add_argument('--output-dir', default='evaluations/hyperparameter_sweep',
                       help='Output directory')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')

    args = parser.parse_args()

    if args.dry_run:
        # Calculate total combinations
        total_combos = len(args.models) * len(args.datasets) * len(args.frame_rates) * \
                      len(args.clip_durations) * len(args.coverages) * len(args.strides)

        print("DRY RUN - Would execute the following:")
        print(f"Total configurations: {total_combos}")
        print(f"Models: {args.models}")
        print(f"Datasets: {args.datasets}")
        print(f"Frame rates: {args.frame_rates}")
        print(f"Clip durations: {args.clip_durations}")
        print(f"Coverages: {args.coverages}")
        print(f"Strides: {args.strides}")
        return

    # Run the sweep
    results = hyperparameter_sweep(
        args.models, args.datasets, args.frame_rates,
        args.clip_durations, args.coverages, args.strides,
        args.max_workers
    )

    # Generate report
    generate_sweep_report(results, args.output_dir)

if __name__ == '__main__':
    main()