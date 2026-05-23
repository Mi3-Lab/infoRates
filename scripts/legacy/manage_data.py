#!/usr/bin/env python3
"""Unified data management CLI

Provides subcommands for building manifests, fixing manifest paths, and downloading datasets.
This consolidates several small scripts into one professional CLI entrypoint.

Usage examples:
  python scripts/manage_data.py build-manifest --clips-dir /path/to/clips --out out.csv
  python scripts/manage_data.py fix-manifest --old-root /old --new-root /new manifest.csv
  python scripts/manage_data.py download --dataset kinetics50 --split train
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ROOT = Path(__file__).parent.parent


def run_cmd(cmd):
    logging.info('Running: %s', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logging.error('Command failed: %s', res.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    logging.info('Done')
    return res


def build_manifest_from_clips(clips_dir, out, split_file=None):
    cmd = [sys.executable, str(ROOT / 'scripts' / 'data_processing' / 'build_manifest_from_clips.py'),
           '--clips-dir', str(clips_dir), '--out', str(out)]
    if split_file:
        cmd += ['--split-file', str(split_file)]
    return run_cmd(cmd)


def fix_manifest_paths(old_root, new_root, *manifests):
    # Prefer direct import for speed and testability; fall back to subprocess if import fails
    try:
        from info_rates.data_utils import replace_manifest_paths
        replace_manifest_paths(old_root, new_root, manifests)
        return None
    except Exception as e:
        # Fallback to running the script as a subprocess
        logging.warning('Direct import failed (%s), falling back to script invocation', e)
        cmd = [sys.executable, str(ROOT / 'scripts' / 'data_processing' / 'fix_manifest_paths.py'),
               str(old_root), str(new_root)] + [str(m) for m in manifests]
        return run_cmd(cmd)


def download_kinetics_subset(split='both'):
    script = ROOT / 'scripts' / 'data_processing' / 'download_kinetics50_subset.sh'
    cmd = [str(script), split]
    return run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description='Data management CLI')
    sub = parser.add_subparsers(dest='cmd')

    b = sub.add_parser('build-manifest', help='Build manifest from clips')
    b.add_argument('--clips-dir', required=True, type=Path)
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--split-file', type=Path, default=None)

    f = sub.add_parser('fix-manifest', help='Fix manifest paths')
    f.add_argument('old_root', type=str)
    f.add_argument('new_root', type=str)
    f.add_argument('manifests', nargs='+', type=Path)

    d = sub.add_parser('download', help='Download dataset or subset')
    d.add_argument('--dataset', required=True, choices=['kinetics50_sub'], help='Dataset to download')
    d.add_argument('--split', choices=['train', 'val', 'both'], default='both')

    args = parser.parse_args()
    if args.cmd == 'build-manifest':
        build_manifest_from_clips(args.clips_dir, args.out, args.split_file)
    elif args.cmd == 'fix-manifest':
        fix_manifest_paths(args.old_root, args.new_root, *args.manifests)
    elif args.cmd == 'download':
        if args.dataset == 'kinetics50_sub':
            download_kinetics_subset(args.split)
        else:
            logging.error('Unknown dataset')
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
