#!/bin/bash
# Script to extract UCF101 RAR archive
# Usage: bash scripts/extract_rar.sh

set -e

RAR_FILE="UCF101-001.rar"
OUT_DIR="UCF101_data"

echo "Checking for unrar/unar tools..."

# Try to find unrar or unar
if command -v unrar &> /dev/null; then
    echo "✓ Found unrar"
    TOOL="unrar"
elif command -v unar &> /dev/null; then
    echo "✓ Found unar"
    TOOL="unar"
elif command -v module &> /dev/null; then
    echo "Trying to load unrar module..."
    module load unrar 2>/dev/null || module load archive-tools 2>/dev/null || true
    if command -v unrar &> /dev/null; then
        echo "✓ Loaded unrar via module"
        TOOL="unrar"
    else
        echo "✗ unrar not available via modules"
        TOOL="none"
    fi
else
    echo "✗ No extraction tool found"
    TOOL="none"
fi

if [ "$TOOL" = "none" ]; then
    echo ""
    echo "ERROR: RAR extraction tool not found!"
    echo "Please install one of:"
    echo "  - unrar: module load unrar  OR  yum install unrar"
    echo "  - unar:  yum install unar"
    exit 1
fi

# Extract the RAR file
if [ -f "$RAR_FILE" ]; then
    echo ""
    echo "Extracting $RAR_FILE to $OUT_DIR..."
    mkdir -p "$OUT_DIR"
    
    if [ "$TOOL" = "unrar" ]; then
        unrar x -o+ "$RAR_FILE" "$OUT_DIR/"
    elif [ "$TOOL" = "unar" ]; then
        unar -o "$OUT_DIR" "$RAR_FILE"
    fi
    
    echo ""
    echo "✓ Extraction complete!"
    echo ""
    echo "Contents of $OUT_DIR:"
    ls -la "$OUT_DIR" | head -20
    
    if [ -d "$OUT_DIR/UCF-101" ]; then
        NUM_CLASSES=$(ls "$OUT_DIR/UCF-101" | wc -l)
        echo ""
        echo "✓ Found UCF-101 with $NUM_CLASSES action classes"
    fi
else
    echo "ERROR: RAR file not found: $RAR_FILE"
    exit 1
fi
