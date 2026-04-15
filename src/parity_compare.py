"""Parity comparison helpers.

Expose functions to load a run manifest and extract canonical metrics.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_metrics_in_artifact(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Discover canonical metrics in a performance-like artifact dict.

    Returns a dict with keys: sharpe, annual_return, annual_volatility, max_drawdown
    Values may be None if not found.
    """
    # candidate key names
    sharpe_keys = ["sharpe", "sharpe_ratio", "annualized_sharpe", "test_sharpe", "train_sharpe", "val_sharpe"]
    ar_keys = ["annual_return", "annualized_return", "cagr", "return_annual", "annualized_return_pct"]
    vol_keys = ["annual_volatility", "annual_vol", "volatility", "annualized_volatility", "std_annual"]
    mdd_keys = ["max_drawdown", "mdd", "max_dd", "drawdown"]

    def pick(keys):
        for k in keys:
            if k in data and isinstance(data[k], (int, float)):
                return float(data[k])
        return None

    return {
        "sharpe": pick(sharpe_keys),
        "annual_return": pick(ar_keys),
        "annual_volatility": pick(vol_keys),
        "max_drawdown": pick(mdd_keys),
    }


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, "manifest.json")
    return _load_json(manifest_path)


def extract_metrics_from_run(manifest_path: str) -> Dict[str, Optional[float]]:
    """Given a manifest.json path (or run dir), scan declared artifacts and common filenames
    and return the first artifact that contains at least one canonical metric. If none found,
    returns a dict of Nones.
    """
    manifest = load_manifest(manifest_path)
    out_dir = os.path.dirname(manifest_path) if os.path.isfile(manifest_path) else manifest_path
    artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []

    candidates = []
    for a in artifacts:
        p = os.path.join(out_dir, f"{a}.json")
        if os.path.exists(p):
            candidates.append(p)
    for name in ("performance.json", "metrics.json", "stats.json"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p) and p not in candidates:
            candidates.append(p)

    # examine candidates
    for p in candidates:
        try:
            d = _load_json(p)
        except Exception:
            continue
        metrics = find_metrics_in_artifact(d)
        # if any metric present, return normalized metrics
        if any(v is not None for v in metrics.values()):
            return metrics

    # nothing found -> return nulls
    return {"sharpe": None, "annual_return": None, "annual_volatility": None, "max_drawdown": None}


def compare_runs(manifest_path_a: str, manifest_path_b: str) -> Dict[str, Any]:
    """Compare two runs (paths to manifest.json or run dirs) and return a normalized diff dict.

    Format:
    {
      "run_a": {metrics...},
      "run_b": {metrics...},
      "diff": {metrics: run_b - run_a (None if missing)}
    }
    """
    a = extract_metrics_from_run(manifest_path_a)
    b = extract_metrics_from_run(manifest_path_b)

    diff = {}
    for k in ("sharpe", "annual_return", "annual_volatility", "max_drawdown"):
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            diff[k] = None
        else:
            diff[k] = vb - va

    return {"run_a": a, "run_b": b, "diff": diff}
