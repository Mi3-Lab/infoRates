#!/bin/bash
# Alternative HMDB-51 Download via academictorrents
# More reliable than direct downloads

set -e

DATA_DIR="HMDB51_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================================================"
echo "Downloading HMDB-51 via Academic Torrents"
echo "========================================================================"

# Install academictorrents client if not available
if ! python -c "import academictorrents" 2>/dev/null; then
    echo "→ Installing academictorrents..."
    pip install -q academictorrents
fi

# Download HMDB-51 using academic torrents
echo "→ Downloading HMDB-51 dataset (~2.16 GB)..."
python << 'PYTHON_EOF'
import academictorrents as at
import os

print("Downloading HMDB-51...")
# HMDB-51 torrent hash
path = at.get("hmdb51", datastore="HMDB51_data")
print(f"Downloaded to: {path}")
PYTHON_EOF

echo ""
echo "========================================================================"
echo "✅ HMDB-51 Download Complete!"
echo "========================================================================"
