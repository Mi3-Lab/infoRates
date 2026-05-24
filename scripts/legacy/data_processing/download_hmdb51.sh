#!/bin/bash
# Download HMDB-51 Dataset
# Source: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

set -e

DATA_DIR="HMDB51_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================================================"
echo "Downloading HMDB-51 Dataset"
echo "========================================================================"
echo "Total size: ~2GB"
echo ""

# Download videos from alternative mirror (original site has moved)
if [ ! -f "hmdb51_org.rar" ]; then
    echo "→ Downloading HMDB-51 videos from alternative source..."
    # Try multiple mirrors
    wget -O hmdb51_org.rar https://www.dropbox.com/s/vz7n5o2y0f4m6k8/hmdb51_org.rar?dl=1 || \
    wget -O hmdb51_org.rar http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_org.rar || \
    wget -O hmdb51_org.rar http://figment.csail.mit.edu/~hueihan/temporal-segment-networks/data/hmdb51_org.rar
else
    echo "✓ hmdb51_org.rar already exists"
fi

# Download train/test splits
if [ ! -f "test_train_splits.rar" ]; then
    echo "→ Downloading train/test splits..."
    wget -O test_train_splits.rar https://www.dropbox.com/s/2dz3k9p0l30f1z6/test_train_splits.rar?dl=1 || \
    wget -O test_train_splits.rar http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/test_train_splits.rar
else
    echo "✓ test_train_splits.rar already exists"
fi

# Check if unrar is installed
if ! command -v unrar &> /dev/null; then
    echo "⚠️  'unrar' not found. Trying to install..."
    if command -v yum &> /dev/null; then
        sudo yum install -y unrar || echo "Please install unrar manually"
    elif command -v apt-get &> /dev/null; then
        sudo apt-get install -y unrar || echo "Please install unrar manually"
    fi
fi

# Extract videos
if [ ! -d "videos" ]; then
    echo "→ Extracting videos..."
    mkdir -p videos
    unrar x hmdb51_org.rar videos/
    
    # Extract nested RAR files (each class is a separate RAR)
    echo "→ Extracting individual class archives..."
    cd videos
    for rar_file in *.rar; do
        if [ -f "$rar_file" ]; then
            class_name="${rar_file%.rar}"
            mkdir -p "$class_name"
            unrar x "$rar_file" "$class_name/"
            rm "$rar_file"  # Clean up after extraction
        fi
    done
    cd ..
else
    echo "✓ Videos already extracted"
fi

# Extract splits
if [ ! -d "splits" ]; then
    echo "→ Extracting train/test splits..."
    mkdir -p splits
    unrar x test_train_splits.rar splits/
else
    echo "✓ Splits already extracted"
fi

echo ""
echo "========================================================================"
echo "✅ HMDB-51 Download Complete!"
echo "========================================================================"
echo "Location: $PWD"
echo "Structure:"
echo "  - videos/ (51 class folders with .avi files)"
echo "  - splits/ (train/test split annotations)"
echo ""

# Count videos
video_count=$(find videos -name "*.avi" 2>/dev/null | wc -l)
echo "Total videos: $video_count"
echo ""
