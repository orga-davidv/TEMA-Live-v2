#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

# Ensure repo src is on sys.path (like other scripts)
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from tema.validation.walkforward import run_walkforward_on_series


def _find_returns_csv(run_path: Path, selector: str) -> Path:
    # selector in {'baseline','ml','ml_meta'}
    mapping = {
        "baseline": "portfolio_test_returns.csv",
        "ml": "portfolio_test_returns_ml.csv",
        "ml_meta": "portfolio_test_returns_ml_meta.csv",
    }
    fname = mapping.get(selector)
    if fname is None:
        raise ValueError(f"unknown selector: {selector}")
    p = run_path / fname
    if p.exists():
        return p
    # fallback: search for pattern
    for f in run_path.iterdir():
        if f.is_file() and f.name.startswith("portfolio_test_returns") and f.suffix == ".csv":
            if selector == "baseline" and f.name == "portfolio_test_returns.csv":
                return f
            if selector == "ml" and "ml" in f.name and "meta" not in f.name:
                return f
            if selector == "ml_meta" and "ml_meta" in f.name:
                return f
    raise FileNotFoundError(f"returns CSV not found for selector {selector} in {run_path}")


def run(path: str, selector: str = "baseline") -> int:
    try:
        p = Path(path)
        if p.is_dir():
            run_dir = p
        else:
            # assume manifest.json path
            if p.exists():
                run_dir = p.parent
            else:
                print(json.dumps({"error": f"path not found: {path}"}))
                return 2
        csv = _find_returns_csv(run_dir, selector)
        df = pd.read_csv(csv)
        # determine column
        col_map = {"baseline": "portfolio_return", "ml": "portfolio_return_ml", "ml_meta": "portfolio_return_ml_meta"}
        col = col_map.get(selector)
        if col not in df.columns:
            # try to infer numeric column besides datetime
            cand = [c for c in df.columns if c != "datetime"]
            if not cand:
                print(json.dumps({"error": "no return column found"}))
                return 2
            col = cand[0]
        df["datetime"] = pd.to_datetime(df["datetime"]) if "datetime" in df.columns else pd.to_datetime(df.iloc[:, 0])
        df = df.set_index("datetime")
        series = df[col].astype(float)

        windows, per_df, summary = run_walkforward_on_series(series)

        # write outputs
        out_dir = run_dir
        per_df.to_csv(out_dir / "walkforward_windows.csv", index=False)
        with open(out_dir / "walkforward_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, default=lambda o: (str(o)), indent=2)

        return 0 if summary.get("passed") else 1
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        return 2
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run walkforward harness on a run directory or manifest")
    parser.add_argument("path", help="Run directory containing outputs or path to manifest.json")
    parser.add_argument("selector", nargs="?", default="baseline", help="Which series to use: baseline|ml|ml_meta")
    args = parser.parse_args()
    sys.exit(run(args.path, args.selector))
