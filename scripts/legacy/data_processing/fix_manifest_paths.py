#!/usr/bin/env python3
import sys
from pathlib import Path

from info_rates.data_utils import replace_manifest_paths


def main():
    if len(sys.argv) < 4:
        print("Usage: fix_manifest_paths.py <old_root> <new_root> <csv1> [<csv2> ...]")
        sys.exit(1)
    old_root = sys.argv[1]
    new_root = sys.argv[2]
    manifests = [Path(p) for p in sys.argv[3:]]
    replace_manifest_paths(old_root, new_root, manifests)


if __name__ == "__main__":
    main()
