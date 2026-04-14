"""
Evaluation harness for baseline/candidate portfolio return series.

Usage (from repo root):
  python3 Template/eval_harness.py \
    --baseline Template/portfolio_test_returns.csv \
    --candidate Template/portfolio_test_returns_ml.csv

The script expects CSVs with columns: datetime,portfolio_return (returns as decimals, e.g. 0.001)
Defaults assume daily data and 252 trading days per year. Change with --periods-per-year.

Outputs a comparison table with total return, annualized return, annualized volatility, Sharpe, and max drawdown.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


def load_returns(path: str, col: str = "portfolio_return") -> pd.Series:
    df = pd.read_csv(path, parse_dates=["datetime"]) if path else pd.DataFrame()
    if df.empty:
        raise ValueError(f"Empty or missing file: {path}")
    if "datetime" not in df.columns:
        raise ValueError(f"CSV {path} must contain a 'datetime' column")
    # flexible column detection: prefer provided col, otherwise try to find a reasonable fallback
    if col not in df.columns:
        # find any column that contains both 'portfolio' and 'return', or any column containing 'return'
        fallback_cols = [c for c in df.columns if 'portfolio' in c.lower() and 'return' in c.lower()]
        if not fallback_cols:
            fallback_cols = [c for c in df.columns if 'return' in c.lower() and c != 'datetime']
        if not fallback_cols:
            raise ValueError(f"CSV {path} must contain a return column (tried '{col}')")
        col = fallback_cols[0]
    s = df.set_index("datetime")[col].sort_index()
    s = s.dropna()
    # ensure numeric
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> Metrics:
    if returns.empty:
        return Metrics(math.nan, math.nan, math.nan, math.nan, math.nan)
    n = returns.shape[0]
    # total return (geometric)
    total_return = (1.0 + returns).prod() - 1.0
    # annualized return (geometric)
    if total_return <= -1.0:
        annual_return = -1.0
    else:
        annual_return = (1.0 + total_return) ** (periods_per_year / max(1, n)) - 1.0
    # annualized volatility
    ann_vol = returns.std(ddof=1) * (periods_per_year ** 0.5)
    # sharpe (assume rf=0)
    sharpe = annual_return / ann_vol if ann_vol and not np.isnan(ann_vol) else math.nan
    # max drawdown
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1.0
    max_dd = drawdown.min()
    return Metrics(total_return, annual_return, ann_vol, sharpe, max_dd)


def compare_metrics(base: Metrics, cand: Metrics) -> pd.DataFrame:
    rows = [
        ("Total Return", base.total_return, cand.total_return),
        ("Annual Return", base.annual_return, cand.annual_return),
        ("Annual Vol", base.annual_vol, cand.annual_vol),
        ("Sharpe", base.sharpe, cand.sharpe),
        ("Max Drawdown", base.max_drawdown, cand.max_drawdown),
    ]
    data = []
    for name, b, c in rows:
        delta = c - b if (not (math.isnan(c) or math.isnan(b))) else math.nan
        # determine better: for Sharpe higher is better; for max drawdown higher (less negative) is better; for vol lower is better; for returns higher is better
        if math.isnan(delta):
            label = "N/A"
        else:
            if name == "Annual Vol":
                label = "better" if c < b else "worse" if c > b else "equal"
            elif name == "Max Drawdown":
                # less negative (=higher) is better
                label = "better" if c > b else "worse" if c < b else "equal"
            else:
                label = "better" if c > b else "worse" if c < b else "equal"
        data.append((name, b, c, delta, label))
    df = pd.DataFrame(data, columns=["metric", "baseline", "candidate", "delta", "candidate_vs_baseline"])
    return df


def fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NaN"
    # format percentages for most metrics
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Candidate evaluation harness")
    parser.add_argument("--baseline", default="Template/portfolio_test_returns.csv", help="baseline CSV path")
    parser.add_argument("--candidate", default="Template/portfolio_test_returns_ml.csv", help="candidate CSV path")
    parser.add_argument("--periods-per-year", type=int, default=252, help="periods per year (default 252)")
    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    base_s = load_returns(args.baseline)
    print(f"Loading candidate: {args.candidate}")
    cand_s = load_returns(args.candidate)

    base_m = compute_metrics(base_s, periods_per_year=args.periods_per_year)
    cand_m = compute_metrics(cand_s, periods_per_year=args.periods_per_year)

    df = compare_metrics(base_m, cand_m)

    # pretty print
    print("\nEvaluation results (values in decimal, e.g. 0.10 = 10%):\n")
    print(df.to_string(index=False, formatters={
        'baseline': lambda x: fmt(x),
        'candidate': lambda x: fmt(x),
        'delta': lambda x: fmt(x),
    }))

    print("\nNotes: Sharpe assumes risk-free = 0. Annualization uses geometric approach based on number of observations and periods-per-year.")


if __name__ == '__main__':
    main()
