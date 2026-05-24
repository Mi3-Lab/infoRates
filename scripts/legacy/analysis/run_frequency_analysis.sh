#!/bin/bash
# Critical Frequency Analysis Script
# Analyzes action frequencies and temporal requirements

echo "=== Critical Frequency Analysis ==="
echo "Analyzing action dynamics across datasets..."

# Create analysis directory
mkdir -p analysis/critical_frequencies

# Run frequency analysis for each dataset
python scripts/analysis/critical_frequency_analysis.py --dataset ucf101
python scripts/analysis/critical_frequency_analysis.py --dataset kinetics400
python scripts/analysis/critical_frequency_analysis.py --dataset hmdb51

echo "Critical frequency analysis complete!"
echo "Results saved to analysis/critical_frequencies/"