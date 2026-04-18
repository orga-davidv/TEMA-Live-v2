import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from tema.backtest import compute_backtest_metrics


def _to_offset(years: int = 0, months: int = 0) -> pd.DateOffset:
    return pd.DateOffset(years=years, months=months)


def generate_walkforward_windows(
    dates: pd.DatetimeIndex,
    train_years: int = 3,
    test_months: int = 6,
    step_months: int = 3,
) -> List[Dict]:
    """Generate walk-forward windows over a DatetimeIndex.

    Each window is a dict with keys: train_start, train_end, test_start, test_end
    All values are pandas.Timestamp. Windows are advanced by `step_months`.
    """
    if len(dates) == 0:
        return []
    train_off = _to_offset(years=train_years)
    test_off = _to_offset(months=test_months)
    step_off = _to_offset(months=step_months)

    windows: List[Dict] = []

    # earliest possible test start is dates[0] + train
    current_test_start = dates[0] + train_off
    last_date = dates[-1]

    while True:
        test_start = current_test_start
        test_end = test_start + test_off
        if test_end > last_date:
            break
        train_start = test_start - train_off
        # align to available dates
        # training: dates >= train_start and < test_start
        train_idx = dates[(dates >= train_start) & (dates < test_start)]
        test_idx = dates[(dates >= test_start) & (dates < test_end)]
        if len(train_idx) and len(test_idx):
            windows.append(
                {
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1],
                }
            )
        # advance
        current_test_start = current_test_start + step_off
        # safety guard to avoid infinite loop
        if len(windows) > 10000:
            break
    return windows


def _build_equity(returns: np.ndarray) -> np.ndarray:
    if returns.size == 0:
        return np.array([])
    return np.cumprod(1.0 + returns)


def compute_window_metrics(
    returns: pd.Series,
    windows: List[Dict],
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute per-window metrics for the test portions of windows.

    Returns a DataFrame with columns: test_start, test_end, sharpe, annual_return, annual_vol, max_drawdown, periods
    """
    rows = []
    for w in windows:
        ts = w["test_start"]
        te = w["test_end"]
        sub = returns.loc[(returns.index >= ts) & (returns.index <= te)]
        r = sub.to_numpy(dtype=float)
        if r.size == 0:
            metrics = {
                "sharpe": float("nan"),
                "annual_return": float("nan"),
                "annual_vol": float("nan"),
                "max_drawdown": float("nan"),
                "periods": 0,
            }
        else:
            eq = _build_equity(r)
            to = np.zeros_like(r)
            m = compute_backtest_metrics(
                r,
                eq,
                to,
                annualization_factor,
                risk_free_rate=float(risk_free_rate),
            )
            metrics = {
                "sharpe": float(m.get("sharpe", float("nan"))),
                "annual_return": float(m.get("annual_return", float("nan"))),
                "annual_vol": float(m.get("annual_vol", float("nan"))),
                "max_drawdown": float(m.get("max_drawdown", float("nan"))),
                "periods": int(m.get("periods", 0)),
            }
        rows.append({"test_start": str(ts), "test_end": str(te), **metrics})
    return pd.DataFrame(rows)


def summarize_metrics(
    per_window_df: pd.DataFrame,
    sharpe_threshold: float = 0.5,
) -> Dict:
    """Summarize per-window metrics.

    Returns dict with median_sharpe, p10_sharpe, worst_max_drawdown, consistency_ratio, pass
    """
    if per_window_df.empty:
        return {
            "median_sharpe": None,
            "p10_sharpe": None,
            "worst_max_drawdown": None,
            "consistency_ratio": 0.0,
            "passed": False,
        }
    sharpes = per_window_df["sharpe"].to_numpy(dtype=float)
    # filter nan
    sharpes = sharpes[np.isfinite(sharpes)]
    mdds = per_window_df["max_drawdown"].to_numpy(dtype=float)
    mdds = mdds[np.isfinite(mdds)]

    median_sharpe = float(np.median(sharpes)) if sharpes.size else None
    p10_sharpe = float(np.percentile(sharpes, 10.0)) if sharpes.size else None
    worst_max_drawdown = float(np.min(mdds)) if mdds.size else None
    consistency_ratio = float(np.sum(sharpes >= sharpe_threshold) / sharpes.size) if sharpes.size else 0.0
    passed = consistency_ratio >= 0.70
    return {
        "median_sharpe": median_sharpe,
        "p10_sharpe": p10_sharpe,
        "worst_max_drawdown": worst_max_drawdown,
        "consistency_ratio": consistency_ratio,
        "passed": bool(passed),
    }


def run_walkforward_on_series(
    returns: pd.Series,
    train_years: int = 3,
    test_months: int = 6,
    step_months: int = 3,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0,
    sharpe_threshold: float = 0.5,
):
    dates = returns.index
    windows = generate_walkforward_windows(dates, train_years=train_years, test_months=test_months, step_months=step_months)
    per_df = compute_window_metrics(
        returns,
        windows,
        annualization_factor=annualization_factor,
        risk_free_rate=risk_free_rate,
    )
    summary = summarize_metrics(per_df, sharpe_threshold=sharpe_threshold)
    return windows, per_df, summary


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run walk-forward harness on a CSV of returns")
    parser.add_argument("csv", help="Path to returns CSV (datetime + return column)")
    parser.add_argument("--column", default=None, help="Column name to use for returns")
    args = parser.parse_args()
    p = args.csv
    if not p or not isinstance(p, str):
        print(json.dumps({"error": "no csv path provided"}))
        sys.exit(2)
    try:
        df = pd.read_csv(p)
        df["datetime"] = pd.to_datetime(df["datetime"]) if "datetime" in df.columns else pd.to_datetime(df.iloc[:, 0])
        df = df.set_index("datetime")
        col = args.column or list(df.columns)[0]
        series = df[col]
        _, per_df, summary = run_walkforward_on_series(series)
        print(json.dumps(summary, default=str))
        sys.exit(0 if summary.get("passed") else 1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(2)
