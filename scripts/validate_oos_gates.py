#!/usr/bin/env python3
"""CLI entrypoint for OOS gates validation.

Wraps src/tema/validation_oos.py to provide a stable script interface under scripts/.
"""
import json
import os
import sys
from argparse import ArgumentParser

# Ensure project src/ is on sys.path so scripts can be executed from the repo root.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tema.validation.oos import validate_oos_gates


def main():
    p = ArgumentParser(description="Validate OOS gates for a run manifest or run directory")
    p.add_argument("path", help="Path to manifest.json or run directory containing manifest.json")
    p.add_argument("--min-sharpe", type=float, default=None)
    p.add_argument("--max-drawdown", type=float, default=None)
    p.add_argument("--max-turnover", type=float, default=None)

    args = p.parse_args()
    path = args.path
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
    else:
        manifest_path = path

    if not os.path.exists(manifest_path):
        print(json.dumps({"error": f"manifest not found: {manifest_path}"}))
        sys.exit(2)

    res = validate_oos_gates(manifest_path, min_sharpe=args.min_sharpe, max_drawdown=args.max_drawdown, max_turnover=args.max_turnover)
    print(json.dumps(res))
    sys.exit(0 if res.get("passed") else 1)


if __name__ == "__main__":
    main()
