#!/usr/bin/env bash
set -euo pipefail

# Download a 50-class subset of Kinetics-400 using precomputed URL lists.
# Requirements: yt-dlp, ffmpeg.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LIST_DIR="$ROOT_DIR/data/Kinetics400_data/k400_50"
OUT_DIR="$ROOT_DIR/data/Kinetics400_data/k400_50"

TRAIN_LIST="$LIST_DIR/train_urls.txt"
VAL_LIST="$LIST_DIR/val_urls.txt"

if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "yt-dlp is required. Install with: pip install yt-dlp" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it via your package manager." >&2
  exit 1
fi

# Usage helpers
usage() {
  cat <<'EOF'
Usage: download_kinetics50_subset.sh [train|val|both]

Reads URL lists from data/Kinetics400_data/k400_50 and downloads trimmed clips
into class-labeled folders under the same directory.

Examples:
  ./scripts/download_kinetics50_subset.sh train   # only training split
  ./scripts/download_kinetics50_subset.sh         # both splits
EOF
}

SPLIT="both"
if [[ $# -gt 1 ]]; then
  usage
  exit 1
elif [[ $# -eq 1 ]]; then
  case "$1" in
    train|val|both) SPLIT="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
fi

process_list() {
  local list_file="$1" split_name="$2"
  if [[ ! -f "$list_file" ]]; then
    echo "Missing list: $list_file" >&2
    return 1
  fi

  echo "Downloading split: $split_name"
  while read -r url label start end; do
    [[ -z "$url" ]] && continue
    local dest_dir="$OUT_DIR/$split_name/$label"
    mkdir -p "$dest_dir"

    # Trim clip to the labeled segment to save space.
    if ! yt-dlp "${url}" \
      --quiet --no-warnings --ignore-errors \
      --download-sections "*${start}-${end}" \
      --merge-output-format mp4 \
      -o "$dest_dir/%(id)s.%(ext)s"; then
      echo "Failed to download ${url} (${label}); skipping" >&2
    fi
  done < "$list_file"
}

if [[ "$SPLIT" == "train" || "$SPLIT" == "both" ]]; then
  process_list "$TRAIN_LIST" train
fi
if [[ "$SPLIT" == "val" || "$SPLIT" == "both" ]]; then
  process_list "$VAL_LIST" val
fi

echo "Done. Clips stored under $OUT_DIR/{train,val}/<class>/"