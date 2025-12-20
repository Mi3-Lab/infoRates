#!/usr/bin/env bash
set -euo pipefail

# Safe cleanup: move old or duplicate UCF101_data outputs to an archive folder.
# Dry-run by default. Pass --apply to perform moves.

APPLY=false
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=true
fi

DATA_ROOT="UCF101_data"
ARCHIVE_DIR="$DATA_ROOT/archive_$(date +%Y%m%d_%H%M%S)"

# Candidates to archive (older test outputs and manifests)
CANDIDATES=(
  "$DATA_ROOT/UCF101_50frames"
  "$DATA_ROOT/UCF101_50frames_test"
  "$DATA_ROOT/test_sample"
  "$DATA_ROOT/test_manifest.csv"
)

# Keep list (do not move)
KEEP=(
  "$DATA_ROOT/UCF-101"                 # original dataset
  "$DATA_ROOT/ucfTrainTestlist"        # official split lists
  "$DATA_ROOT/UCF101_fixed_50f"        # current fixed clips
  "$DATA_ROOT/ucf101_manifest.csv"     # original video manifest
  "$DATA_ROOT/ucf101_fixedlen_50f.csv" # fixed clips manifest
  "$DATA_ROOT/results_fixedlen_finetuned.csv" # eval results
)

echo "Inspecting $DATA_ROOT ..."

# Show sizes
echo "\nDirectory sizes (top-level):"
du -sh "$DATA_ROOT"/* 2>/dev/null | sort -h || true

# Show clip counts in candidates
echo "\nClip counts in candidate folders:"
for d in "${CANDIDATES[@]}"; do
  if [[ -d "$d" ]]; then
    echo "$d"
    find "$d" -type f \( -name '*.mp4' -o -name '*.avi' -o -name '*.npy' \) | wc -l
  fi
done

# Plan moves
TO_MOVE=()
for d in "${CANDIDATES[@]}"; do
  if [[ -e "$d" ]]; then
    TO_MOVE+=("$d")
  fi
done

if [[ ${#TO_MOVE[@]} -eq 0 ]]; then
  echo "\nNo candidate folders/files found to archive."
  exit 0
fi

echo "\nPlanned moves to archive:"
for p in "${TO_MOVE[@]}"; do
  echo " - $p -> $ARCHIVE_DIR/"
done

if [[ "$APPLY" == "true" ]]; then
  echo "\nApplying moves..."
  mkdir -p "$ARCHIVE_DIR"
  for p in "${TO_MOVE[@]}"; do
    mv "$p" "$ARCHIVE_DIR/"
  done
  echo "Done. Archived to: $ARCHIVE_DIR"
else
  echo "\nDry-run. To apply, rerun: bash scripts/cleanup_ucf101.sh --apply"
fi
