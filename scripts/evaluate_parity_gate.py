#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parity_compare import DEFAULT_PARITY_THRESHOLDS, evaluate_parity_thresholds  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate parity comparison JSON against explicit thresholds")
    parser.add_argument("comparison_json", help="Path to parity_metrics_comparison.json")
    parser.add_argument("--threshold-sharpe", type=float, default=DEFAULT_PARITY_THRESHOLDS["sharpe"])
    parser.add_argument("--threshold-annual-return", type=float, default=DEFAULT_PARITY_THRESHOLDS["annual_return"])
    parser.add_argument("--threshold-annual-volatility", type=float, default=DEFAULT_PARITY_THRESHOLDS["annual_volatility"])
    parser.add_argument("--threshold-max-drawdown", type=float, default=DEFAULT_PARITY_THRESHOLDS["max_drawdown"])
    args = parser.parse_args()

    comparison_path = Path(args.comparison_json)
    with open(comparison_path, "r", encoding="utf-8") as fh:
        comparison = json.load(fh)

    gate = evaluate_parity_thresholds(
        comparison,
        thresholds={
            "sharpe": args.threshold_sharpe,
            "annual_return": args.threshold_annual_return,
            "annual_volatility": args.threshold_annual_volatility,
            "max_drawdown": args.threshold_max_drawdown,
        },
    )
    print(json.dumps(gate))
    return 0 if bool(gate.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
