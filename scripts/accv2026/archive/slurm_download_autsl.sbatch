#!/usr/bin/env bash
#SBATCH --job-name=accv26-download-autsl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=0-05:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

AUTSL_DIR="/data/wesleyferreiramaia/infoRates/data/AUTSL_data"
mkdir -p "$AUTSL_DIR"

echo "======================================================="
echo "AUTSL download via curl — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

if [[ -f "$AUTSL_DIR/.download_complete" ]]; then
    echo "AUTSL already downloaded — skipping"
    exit 0
fi

ZIP="$AUTSL_DIR/autsl.zip"

if [[ ! -f "$ZIP" || ! -s "$ZIP" ]]; then
    echo "[1/2] Downloading AUTSL zip (~13GB) via curl..."
    curl -L \
        -u "wesleymaia999:KGAT_43582f650dccbbb4d8d2c8af6a1199d3" \
        "https://www.kaggle.com/api/v1/datasets/download/sttaseen/autsl" \
        -o "$ZIP" \
        --progress-bar \
        --retry 3 --retry-delay 10
    echo "[1/2] Download done: $(ls -lh $ZIP)"
else
    echo "[1/2] Zip already exists: $(ls -lh $ZIP)"
fi

echo "[2/2] Unzipping AUTSL..."
unzip -q "$ZIP" -d "$AUTSL_DIR/"
touch "$AUTSL_DIR/.download_complete"

echo "======================================================="
echo "AUTSL done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Items: $(ls $AUTSL_DIR | wc -l)"
echo "  Size:  $(du -sh $AUTSL_DIR | cut -f1)"
echo "======================================================="
