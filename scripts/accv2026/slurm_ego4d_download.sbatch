#!/usr/bin/env bash
#SBATCH --job-name=ego4d-download
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -uo pipefail
echo "=== Ego4D Download + Preprocess — $(date) ==="

EGO4D_ROOT="/scratch/wesleyferreiramaia/infoRates/data/Ego4D_data"
REPO="/data/wesleyferreiramaia/infoRates"

source "${REPO}/.venv/bin/activate"

CLIP_UID_FILE="${REPO}/evaluations/accv2026/ego4d_clip_uids.txt"
echo "--- Step 1: Download clips subset (288 clips, ~12 GB) ---"
echo "Using clip UID file: ${CLIP_UID_FILE} ($(wc -l < "${CLIP_UID_FILE}") clips)"
ego4d -o "${EGO4D_ROOT}" --datasets clips --benchmarks FHO \
  --video_uid_file "${CLIP_UID_FILE}" -y

echo ""
echo "--- Step 2: Extract action segments ---"
python "${REPO}/scripts/accv2026/preprocess_ego4d.py" \
    --data-root "${EGO4D_ROOT}" \
    --workers 16 \
    --min-examples-per-class 10

echo ""
echo "--- Step 3: Create 20-per-class eval manifest ---"
python -c "
import pandas as pd
from pathlib import Path

manifest = Path('${EGO4D_ROOT}/splits/ego4d_manifest.csv')
df = pd.read_csv(manifest)
out_dir = Path('${REPO}/evaluations/accv2026/manifests')
out_dir.mkdir(parents=True, exist_ok=True)

# 20 per class for val
val = df[df['split'] == 'val']
sampled = val.groupby('label').apply(lambda g: g.sample(min(20, len(g)), random_state=42)).reset_index(drop=True)
out = out_dir / 'ego4d_val_20_per_class.csv'
sampled.to_csv(out, index=False)
print(f'Eval manifest: {len(sampled)} samples, {sampled[\"label\"].nunique()} classes → {out}')
"

echo ""
echo "=== Ego4D pipeline complete — $(date) ==="
