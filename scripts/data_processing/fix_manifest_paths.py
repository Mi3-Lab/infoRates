#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

def fix_csv(path: Path, old_root: str, new_root: str, column: str = "video_path"):
    df = pd.read_csv(path)
    if column in df.columns:
        df[column] = df[column].astype(str).str.replace(old_root, new_root, regex=False)
        df.to_csv(path, index=False)
        print(f"✓ Updated {path} ({old_root} -> {new_root})")
    else:
        print(f"! Skipped {path}: missing column '{column}'")

def main():
    if len(sys.argv) < 4:
        print("Usage: fix_manifest_paths.py <old_root> <new_root> <csv1> [<csv2> ...]")
        sys.exit(1)
    old_root = sys.argv[1]
    new_root = sys.argv[2]
    for p in sys.argv[3:]:
        fix_csv(Path(p), old_root, new_root)

if __name__ == "__main__":
    main()
