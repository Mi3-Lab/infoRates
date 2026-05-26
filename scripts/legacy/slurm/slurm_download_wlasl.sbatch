#!/usr/bin/env bash
#SBATCH --job-name=accv26-download-wlasl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates

WLASL_DIR="/scratch/wesleyferreiramaia/infoRates/data/WLASL_data"
VIDEOS="${WLASL_DIR}/raw_videos"
HF_BASE="https://huggingface.co/datasets/aipieces/WLASL/resolve/main"

mkdir -p "${WLASL_DIR}" "${VIDEOS}"
echo "WLASL download — $(date)"

# Download annotation JSON
JSON="${WLASL_DIR}/WLASL_v0.3.json"
if [[ ! -f "${JSON}" ]]; then
    echo "Downloading WLASL_v0.3.json..."
    curl -fL --retry 5 --retry-delay 10 \
        "${HF_BASE}/WLASL_v0.3.json" \
        -o "${JSON}"
    echo "  -> $(du -sh "${JSON}" | cut -f1)"
fi

# Download + extract 14 shards
for i in $(seq 1 14); do
    SHARD=$(printf "shard_%03d_014.zip" "$i")
    URL="${HF_BASE}/start_kit/raw_videos_mp4/${SHARD}"
    ZIP="${WLASL_DIR}/${SHARD}"

    if [[ ! -f "${ZIP}" ]]; then
        echo "Downloading ${SHARD}..."
        curl -fL --retry 5 --retry-delay 10 "${URL}" -o "${ZIP}" --progress-bar
        echo "  -> $(du -sh "${ZIP}" | cut -f1)"
    fi

    echo "Extracting ${SHARD}..."
    if unzip -q -n "${ZIP}" "*.mp4" -d "${VIDEOS}/" 2>/dev/null; then
        echo "  extracted directly"
    else
        TMP="${WLASL_DIR}/_tmp_${i}"
        mkdir -p "${TMP}"
        unzip -q -n "${ZIP}" -d "${TMP}/" 2>/dev/null || true
        find "${TMP}/" -name "*.mp4" -exec mv {} "${VIDEOS}/" \; 2>/dev/null || true
        rm -rf "${TMP}/"
    fi

    rm -f "${ZIP}"
done

N=$(find "${VIDEOS}" -name "*.mp4" | wc -l)
echo ""
echo "Done! ${N} MP4 files in ${VIDEOS}"
echo "$(du -sh "${WLASL_DIR}")"
