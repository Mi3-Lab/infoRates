#!/usr/bin/env bash
# After training completes, scan per-epoch checkpoint metadata to find the best epoch,
# then copy it to the final checkpoint path (overwriting the last-epoch checkpoint).
#
# Usage: bash scripts/accv2026/fix_best_checkpoint.sh <checkpoint_base_path>
# Example: bash scripts/accv2026/fix_best_checkpoint.sh \
#              fine_tuned_models/accv2026_videomae_autsl_full_e10_h200
set -euo pipefail

BASE="$1"
if [[ -z "$BASE" ]]; then
  echo "Usage: $0 <checkpoint_base_path>"
  exit 1
fi

if [[ ! -d "$BASE" ]]; then
  echo "Final checkpoint dir not found: $BASE"
  echo "Will create it from the best epoch checkpoint."
fi

source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=src

python3 - << PYEOF
import json, shutil, sys
from pathlib import Path

base = Path("$BASE")
parent = base.parent
name = base.name

# Find all epoch checkpoints
epoch_dirs = sorted(parent.glob(f"{name}_epoch*"))
if not epoch_dirs:
    print(f"No epoch checkpoints found matching {parent}/{name}_epoch*")
    sys.exit(1)

best_acc = -1.0
best_dir = None
best_epoch = -1

for ep_dir in epoch_dirs:
    meta_file = ep_dir / "accv_meta.json"
    if not meta_file.exists():
        continue
    with open(meta_file) as f:
        meta = json.load(f)
    val_acc = meta.get("val_acc", meta.get("best_val_acc", -1))
    epoch   = meta.get("epoch", -1)
    print(f"  Epoch {epoch:>2d}: val_acc={val_acc:.4f}  [{ep_dir.name}]")
    if val_acc > best_acc:
        best_acc  = val_acc
        best_epoch = epoch
        best_dir  = ep_dir

if best_dir is None:
    print("Could not determine best epoch (no accv_meta.json with val_acc found)")
    sys.exit(1)

print(f"\nBest: epoch {best_epoch}, val_acc={best_acc:.4f}  ({best_dir})")

if base.exists():
    # Check if current final checkpoint is already the best
    curr_meta = base / "accv_meta.json"
    if curr_meta.exists():
        with open(curr_meta) as f:
            curr = json.load(f)
        curr_acc = curr.get("val_acc", curr.get("best_val_acc", -1))
        if curr.get("is_best") or curr_acc >= best_acc - 1e-6:
            print(f"Final checkpoint already correct (val_acc={curr_acc:.4f}). Nothing to do.")
            sys.exit(0)
    print(f"Overwriting final checkpoint {base} with best epoch {best_epoch}...")
    shutil.rmtree(base)

shutil.copytree(best_dir, base)
# Update metadata to record this is the best checkpoint
with open(base / "accv_meta.json") as f:
    meta = json.load(f)
meta["is_best"] = True
meta["fixed_from_epoch"] = best_epoch
with open(base / "accv_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Done. {base} now contains epoch {best_epoch} (val_acc={best_acc:.4f})")
PYEOF
