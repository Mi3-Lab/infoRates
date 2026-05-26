#!/usr/bin/env bash
#SBATCH --job-name=accv26-download
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export KAGGLE_USERNAME=wesleymaia999
export KAGGLE_KEY=KGAT_43582f650dccbbb4d8d2c8af6a1199d3

pip install kaggle -q 2>/dev/null || true

echo "======================================================="
echo "Dataset downloads — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ---- AUTSL (Turkish Sign Language, 226 classes, ~38k videos, 13GB) ----
AUTSL_DIR="/data/wesleyferreiramaia/infoRates/data/AUTSL_data"
mkdir -p "$AUTSL_DIR"

if [[ ! -f "$AUTSL_DIR/.download_complete" ]]; then
  echo "[1/2] Downloading AUTSL (~13GB)..."
  kaggle datasets download -d sttaseen/autsl -p "$AUTSL_DIR/" --unzip
  touch "$AUTSL_DIR/.download_complete"
  echo "[1/2] AUTSL done — $(ls $AUTSL_DIR | wc -l) items"
else
  echo "[1/2] AUTSL already downloaded"
fi

# ---- Drive&Act (in-cabin activities, 83 classes, 5.7GB) ----
DRIVEACT_DIR="/data/wesleyferreiramaia/infoRates/data/DriveAct_data"
mkdir -p "$DRIVEACT_DIR"

if [[ ! -f "$DRIVEACT_DIR/.download_complete" ]]; then
  echo "[2/2] Downloading Drive&Act (~5.7GB)..."
  kaggle datasets download -d vafaeii/drive-and-act-rgb -p "$DRIVEACT_DIR/" --unzip
  touch "$DRIVEACT_DIR/.download_complete"
  echo "[2/2] Drive&Act done — $(ls $DRIVEACT_DIR | wc -l) items"
else
  echo "[2/2] Drive&Act already downloaded"
fi

echo "======================================================="
echo "Downloads complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "AUTSL:    $(du -sh $AUTSL_DIR 2>/dev/null | cut -f1)"
echo "Drive&Act: $(du -sh $DRIVEACT_DIR 2>/dev/null | cut -f1)"
echo "======================================================="
