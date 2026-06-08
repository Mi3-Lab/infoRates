#!/usr/bin/env bash
#SBATCH --job-name=accv26-download
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/mnt/datasets/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/mnt/datasets/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /mnt/datasets/infoRates
mkdir -p evaluations/accv2026/logs

export KAGGLE_USERNAME=wesleymaia999
export KAGGLE_KEY=KGAT_43582f650dccbbb4d8d2c8af6a1199d3

pip install kaggle ego4d -q 2>/dev/null || true

echo "======================================================="
echo "Dataset downloads — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

# ---- AUTSL (Turkish Sign Language) ----
AUTSL_DIR="data/AUTSL_data"
mkdir -p "$AUTSL_DIR"

if [[ ! -f "$AUTSL_DIR/.download_complete" ]]; then
  echo "[1/5] Downloading AUTSL (~13GB)..."
  kaggle datasets download -d sttaseen/autsl -p "$AUTSL_DIR/" --unzip
  touch "$AUTSL_DIR/.download_complete"
else
  echo "[1/5] AUTSL already downloaded"
fi

# ---- Drive&Act ----
DRIVEACT_DIR="data/DriveAct_data"
mkdir -p "$DRIVEACT_DIR"

if [[ ! -f "$DRIVEACT_DIR/.download_complete" ]]; then
  echo "[2/5] Downloading Drive&Act (~5.7GB)..."
  kaggle datasets download -d vafaeii/drive-and-act-rgb -p "$DRIVEACT_DIR/" --unzip
  touch "$DRIVEACT_DIR/.download_complete"
else
  echo "[2/5] Drive&Act already downloaded"
fi

# ---- FLAME (Fire/Smoke) ----
FLAME_DIR="data/FLAME_data"
mkdir -p "$FLAME_DIR/raw_archives"

if [[ ! -f "$FLAME_DIR/.download_complete" ]]; then
  echo "[3/5] Downloading FLAME (~0.8GB)..."
  kaggle datasets download -d brycehopkins/flame -p "$FLAME_DIR/raw_archives/" --unzip
  touch "$FLAME_DIR/.download_complete"
else
  echo "[3/5] FLAME already downloaded"
fi

# ---- UCF-Crime (Surveillance) ----
UCF_CRIME_DIR="data/UCFCrime_data"
mkdir -p "$UCF_CRIME_DIR/raw_archives"

if [[ ! -f "$UCF_CRIME_DIR/.download_complete" ]]; then
  echo "[4/5] Downloading UCF-Crime (~128GB)..."
  # Using the itssunny/ucf-crime version as it is the most complete video set
  kaggle datasets download -d itssunny/ucf-crime -p "$UCF_CRIME_DIR/raw_archives/" --unzip
  touch "$UCF_CRIME_DIR/.download_complete"
else
  echo "[4/5] UCF-Crime already downloaded"
fi

# ---- Ego4D ----
EGO4D_DIR="data/Ego4D_data"
mkdir -p "$EGO4D_DIR"

if [[ ! -f "$EGO4D_DIR/.download_complete" ]]; then
  echo "[5/5] Ego4D requires AWS credentials. Skipping automatic download."
  echo "Please run: aws configure && ego4d --output_directory $EGO4D_DIR --datasets annotations clips"
else
  echo "[5/5] Ego4D already downloaded"
fi

echo "======================================================="
echo "Downloads complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "AUTSL:     $(du -sh $AUTSL_DIR 2>/dev/null | cut -f1)"
echo "Drive&Act: $(du -sh $DRIVEACT_DIR 2>/dev/null | cut -f1)"
echo "FLAME:     $(du -sh $FLAME_DIR 2>/dev/null | cut -f1)"
echo "UCF-Crime: $(du -sh $UCF_CRIME_DIR 2>/dev/null | cut -f1)"
echo "======================================================="
