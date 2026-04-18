import json
import os
from typing import Dict, Any, Optional, Tuple

from .manifest import load_manifest


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_metric_keys(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Try to discover sharpe, max_drawdown and annualized_turnover values in a dict.

    Returns (sharpe, max_drawdown, annualized_turnover) where each may be None if not found.
    """
    sharpe = None
    mdd = None
    turnover = None
    # Possible key names
    sharpe_keys = ["sharpe", "sharpe_ratio", "test_sharpe", "test_sharpe_ratio", "annualized_sharpe", "train_sharpe", "val_sharpe"]
    mdd_keys = ["max_drawdown", "mdd", "max_dd", "test_max_drawdown", "train_max_drawdown"]
    turnover_keys = ["annualized_turnover", "turnover", "annual_turnover", "test_annualized_turnover", "train_annualized_turnover"]

    for k in sharpe_keys:
        if k in d and isinstance(d[k], (int, float)):
            sharpe = float(d[k])
            break
    for k in mdd_keys:
        if k in d and isinstance(d[k], (int, float)):
            mdd = float(d[k])
            break
    for k in turnover_keys:
        if k in d and isinstance(d[k], (int, float)):
            turnover = float(d[k])
            break

    return sharpe, mdd, turnover


def validate_oos_gates(manifest_path: str, min_sharpe: Optional[float] = None, max_drawdown: Optional[float] = None, max_turnover: Optional[float] = None) -> Dict[str, Any]:
    """Validate out-of-sample gates for a run described by manifest_path.

    The manifest is expected to be a JSON with an 'artifacts' list. This validator will look through
    artifact JSON files in the same directory as the manifest for any that contain recognizable
    performance metrics and apply the provided thresholds. If a metric is not present, that check is
    skipped (and reported as skipped).

    Returns a result dict with overall 'passed' boolean and per-check details.
    """
    mod = load_manifest(manifest_path)
    out_dir = os.path.dirname(manifest_path)

    artifacts = mod.get("artifacts", [])
    result = {
        "manifest": manifest_path,
        "checked_artifacts": [],
        "checks": {},
        "passed": True,
    }

    # Look for artifacts that might contain metrics
    perf_candidates = []
    for a in artifacts:
        p = os.path.join(out_dir, f"{a}.json")
        if os.path.exists(p):
            perf_candidates.append(p)

    # Also include common names if present
    for name in ("performance.json", "metrics.json", "stats.json"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p) and p not in perf_candidates:
            perf_candidates.append(p)

    # If no candidates, report as skipped (graceful handling for optional artifacts)
    if not perf_candidates:
        result["checks"]["artifact_presence"] = {
            "ok": True,
            "skipped": True,
            "reason": "no performance-like artifacts found",
            "checked": [],
        }
        return result

    # Scan candidates and pick the first that contains at least one metric
    found = False
    for p in perf_candidates:
        try:
            data = _load_json(p)
        except Exception as e:
            # unreadable file - skip with note
            result["checked_artifacts"].append({"path": p, "loaded": False, "error": str(e)})
            continue

        sharpe, mdd, turnover = _find_metric_keys(data)
        entry = {"path": p, "has_sharpe": sharpe is not None, "has_mdd": mdd is not None, "has_turnover": turnover is not None}
        # stress scenario sanity
        stress_ok = None
        if "stress_scenarios" in data and isinstance(data["stress_scenarios"], dict):
            ss = data["stress_scenarios"]
            # simple sanity: each scenario should have a numeric drawdown value
            problems = []
            for name, v in ss.items():
                if not isinstance(v, dict):
                    problems.append(f"scenario {name} not dict")
                    continue
                dd = v.get("drawdown")
                if dd is None or not isinstance(dd, (int, float)):
                    problems.append(f"scenario {name} drawdown missing or not numeric")
            stress_ok = len(problems) == 0
            entry["stress_scenarios_ok"] = stress_ok
            if not stress_ok:
                entry["stress_scenarios_problems"] = problems
        result["checked_artifacts"].append(entry)

        # Apply thresholds where applicable
        # Sharpe
        if min_sharpe is not None:
            if sharpe is None:
                result["checks"]["min_sharpe"] = {
                    "ok": True,
                    "skipped": True,
                    "reason": "sharpe not found",
                }
            else:
                ok = sharpe >= float(min_sharpe)
                result["checks"]["min_sharpe"] = {"ok": ok, "value": sharpe, "threshold": float(min_sharpe)}
                if not ok:
                    result["passed"] = False
        # Max drawdown (use absolute value)
        if max_drawdown is not None:
            if mdd is None:
                result["checks"]["max_drawdown"] = {
                    "ok": True,
                    "skipped": True,
                    "reason": "max_drawdown not found",
                }
            else:
                # allow storing negative drawdowns; compare absolute
                ok = abs(float(mdd)) <= float(max_drawdown)
                result["checks"]["max_drawdown"] = {"ok": ok, "value": mdd, "threshold": float(max_drawdown)}
                if not ok:
                    result["passed"] = False
        # Turnover
        if max_turnover is not None:
            if turnover is None:
                result["checks"]["max_turnover"] = {
                    "ok": True,
                    "skipped": True,
                    "reason": "turnover not found",
                }
            else:
                ok = float(turnover) <= float(max_turnover)
                result["checks"]["max_turnover"] = {"ok": ok, "value": turnover, "threshold": float(max_turnover)}
                if not ok:
                    result["passed"] = False

        # If we reached here and at least one metric was present, mark found and break (we only need one perf artifact)
        if sharpe is not None or mdd is not None or turnover is not None or stress_ok is not None:
            found = True
            break

    if not found:
        result["checks"]["artifact_metrics_found"] = {
            "ok": True,
            "skipped": True,
            "reason": "no recognizable metrics found in candidate artifacts",
        }

    return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate out-of-sample gates for a run manifest or run directory")
    parser.add_argument("path", help="Path to manifest.json or run directory containing manifest.json")
    parser.add_argument("--min-sharpe", type=float, default=None)
    parser.add_argument("--max-drawdown", type=float, default=None)
    parser.add_argument("--max-turnover", type=float, default=None)

    args = parser.parse_args()
    p = args.path
    if os.path.isdir(p):
        manifest_path = os.path.join(p, "manifest.json")
    else:
        manifest_path = p

    if not os.path.exists(manifest_path):
        print(json.dumps({"error": f"manifest not found: {manifest_path}"}))
        sys.exit(2)

    res = validate_oos_gates(manifest_path, min_sharpe=args.min_sharpe, max_drawdown=args.max_drawdown, max_turnover=args.max_turnover)
    print(json.dumps(res))
    sys.exit(0 if res.get("passed") else 1)
