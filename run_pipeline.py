"""Small CLI to run either the modular Wave 2 pipeline or the legacy monolith.

Usage:
  python run_pipeline.py [--run-id ID] [--legacy]

If --legacy is provided we execute Template/TEMA-TEMPLATE(NEW_).py via runpy.
Otherwise we call src.tema.pipeline.run_pipeline.

This script is intentionally minimal and deterministic so CI/tests can exercise both
paths without touching project-wide configuration.
"""
import argparse
import sys
import os
import runpy
from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parent
# Ensure src is on sys.path so "tema" package can be imported
sys.path.insert(0, str(ROOT / "src"))


def run_legacy(run_id: str, out_root: str = "outputs"):
    """Run the legacy monolith only when the env var TEMA_RUN_LEGACY_EXECUTE=1 is set.

    By default this function will create a best-effort manifest and NOT execute the
    legacy script. This keeps the CLI safe and deterministic for CI/tests while still
    providing an explicit opt-in to run the old monolith.
    """
    legacy_path = ROOT / "Template" / "TEMA-TEMPLATE(NEW_).py"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy monolith not found: {legacy_path}")

    should_exec = os.environ.get("TEMA_RUN_LEGACY_EXECUTE", "0") == "1"
    # sanitize run_id to avoid path traversal
    if not re.match(r'^[A-Za-z0-9._-]+$', run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")

    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mf = out_dir / "manifest.json"
    if should_exec:
        # run in its own globals to emulate script execution
        g = {"__name__": "__main__", "RUN_ID": run_id, "OUT_ROOT": out_root}
        runpy.run_path(str(legacy_path), run_name="__main__", init_globals=g)
        # write manifest safely as JSON
        with open(mf, 'w', encoding='utf-8') as fh:
            json.dump({"run_id": run_id, "legacy_executed": True}, fh, indent=2)
    else:
        # do not execute by default; record that we skipped execution
        with open(mf, 'w', encoding='utf-8') as fh:
            json.dump({"run_id": run_id, "legacy_executed": False, "note": "set TEMA_RUN_LEGACY_EXECUTE=1 to actually run the legacy script"}, fh, indent=2)

    return {'manifest_path': str(mf), 'out_dir': str(out_dir)}


def run_modular(run_id: str, out_root: str = "outputs"):
    from tema.pipeline import run_pipeline as rp
    return rp(run_id=run_id, out_root=out_root)


def main(argv=None):
    p = argparse.ArgumentParser("run_pipeline")
    p.add_argument("--run-id", default="manual-run")
    p.add_argument("--legacy", action="store_true")
    args = p.parse_args(argv)

    if args.legacy:
        res = run_legacy(args.run_id)
    else:
        res = run_modular(args.run_id)
    print(res)
    return res


if __name__ == "__main__":
    main()
