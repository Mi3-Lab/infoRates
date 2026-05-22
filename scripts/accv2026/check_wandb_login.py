#!/usr/bin/env python3
"""Fail fast unless W&B has usable credentials."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import wandb
    except ImportError:
        print("[wandb-check] wandb is not installed.", file=sys.stderr)
        return 2

    try:
        logged_in = wandb.login(verify=True)
    except Exception as exc:
        print(f"[wandb-check] W&B login verification failed: {exc}", file=sys.stderr)
        return 2

    if not logged_in:
        print("[wandb-check] W&B is not logged in. Run: wandb login", file=sys.stderr)
        return 2

    print("[wandb-check] W&B login verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

