#!/bin/bash
# Download Kinetics-400 Mini (50 classes subset)
# Source: https://github.com/hassony2/kinetics_400_mini

set -e

DATA_DIR="Kinetics400_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================================================"
echo "Downloading Kinetics-400 Mini Dataset"
echo "========================================================================"
echo "This will download ~50 classes from Kinetics-400"
echo "Total size: ~20-30GB"
echo ""

# Clone the downloader repository
if [ ! -d "kinetics_downloader" ]; then
    echo "→ Cloning downloader..."
    git clone https://github.com/hassony2/kinetics_downloader.git
fi

cd kinetics_downloader

# Install required packages
echo "→ Installing dependencies..."
pip install -q pandas youtube-dl tqdm

# Download train split (50 classes)
echo "→ Downloading training videos (this will take a while)..."
python download.py --classes 50 --split train --output_dir ../train

# Download validation split
echo "→ Downloading validation videos..."
python download.py --classes 50 --split val --output_dir ../val

cd ..

echo ""
echo "========================================================================"
echo "✅ Kinetics-400 Mini Download Complete!"
echo "========================================================================"
echo "Location: $PWD"
echo "Structure:"
echo "  - train/ (training videos)"
echo "  - val/ (validation videos)"
echo ""
echo "Note: Some videos may fail to download (deleted from YouTube)"
echo "This is normal - the dataset will still be usable"
