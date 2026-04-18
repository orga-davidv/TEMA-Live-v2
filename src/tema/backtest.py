from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from .execution.costs import compute_transaction_cost


def _annualization_factor(freq: str) -> float:
    mapping = {
        "D": 252.0,
        "H": 252.0 * 24.0,
        "W": 52.0,
        "M": 12.0,
    }
    return float(mapping.get(str(freq).upper(), 252.0))


def _normalize_row_to_weights(row: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(row.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    long_only = np.clip(x, 0.0, None)
    if float(np.sum(long_only)) > 0.0:
        return long_only / float(np.sum(long_only))
    abs_sum = float(np.sum(np.abs(x)))
    if abs_sum > 0.0:
        return np.abs(x) / abs_sum
    return fallback.copy()


def build_weight_schedule_from_signals(signal_df: pd.DataFrame, fallback_weights: Sequence[float]) -> np.ndarray:
    fallback = np.asarray(fallback_weights, dtype=float)
    if fallback.ndim != 1:
        raise ValueError("fallback_weights must be one-dimensional")
    if signal_df.empty:
        return np.empty((0, len(fallback)), dtype=float)
    if signal_df.shape[1] != len(fallback):
        raise ValueError("signal_df columns must match fallback_weights length")
    out = np.zeros((len(signal_df), signal_df.shape[1]), dtype=float)
    arr = signal_df.to_numpy(dtype=float)
    for i in range(arr.shape[0]):
        out[i] = _normalize_row_to_weights(arr[i], fallback)
    return out


@dataclass
class BacktestResult:
    periodic_returns: list[float]
    equity_curve: list[float]
    turnover_series: list[float]
    annualization_factor: float
    metrics: dict


def run_return_equity_simulation(
    asset_returns: np.ndarray,
    target_weights: np.ndarray,
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    cost_model: str = "simple",
    spread_bps: float = 0.0,
    impact_coeff: float = 0.0,
    borrow_bps: float = 0.0,
    freq: str = "D",
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    ret = np.asarray(asset_returns, dtype=float)
    w = np.asarray(target_weights, dtype=float)
    if ret.ndim != 2 or w.ndim != 2:
        raise ValueError("asset_returns and target_weights must be 2D arrays [time, assets]")
    if ret.shape != w.shape:
        raise ValueError("asset_returns and target_weights must have identical shapes")
    if ret.shape[0] == 0:
        ann = _annualization_factor(freq)
        metrics = compute_backtest_metrics(
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            ann,
            risk_free_rate=risk_free_rate,
        )
        return BacktestResult([], [], [], ann, metrics)

    # Walk-forward-friendly execution: apply previous target weights to current period return.
    executed = np.vstack([w[0], w[:-1]])
    periodic_returns = np.zeros(ret.shape[0], dtype=float)
    turnover_series = np.zeros(ret.shape[0], dtype=float)
    prev = executed[0]
    for t in range(ret.shape[0]):
        cur = executed[t]
        turnover = float(np.sum(np.abs(cur - prev)))
        pnl = float(np.dot(cur, ret[t]))
        # compute transaction cost using the configured cost model
        cost = compute_transaction_cost(
            turnover,
            prev,
            cur,
            exposure=cur,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            cost_model=cost_model,
            spread_bps=spread_bps,
            impact_coeff=impact_coeff,
            borrow_bps=borrow_bps,
        )

        periodic_returns[t] = pnl - float(cost)
        turnover_series[t] = turnover
        prev = cur
    equity = np.cumprod(1.0 + periodic_returns)
    ann = _annualization_factor(freq)
    metrics = compute_backtest_metrics(
        periodic_returns,
        equity,
        turnover_series,
        ann,
        risk_free_rate=risk_free_rate,
    )
    return BacktestResult(
        periodic_returns=periodic_returns.tolist(),
        equity_curve=equity.tolist(),
        turnover_series=turnover_series.tolist(),
        annualization_factor=ann,
        metrics=metrics,
    )


def compute_backtest_metrics(
    periodic_returns: np.ndarray,
    equity_curve: np.ndarray,
    turnover_series: np.ndarray,
    annualization_factor: float,
    risk_free_rate: float = 0.0,
) -> dict:
    r = np.asarray(periodic_returns, dtype=float)
    e = np.asarray(equity_curve, dtype=float)
    to = np.asarray(turnover_series, dtype=float)
    if r.size == 0:
        return {
            "sharpe": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "max_drawdown": 0.0,
            "annualized_turnover": 0.0,
            "turnover_proxy": 0.0,
            "periods": 0,
        }
    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=0))
    annual_vol = std_r * float(np.sqrt(annualization_factor))
    gross = float(np.prod(1.0 + r))
    annual_return = float(gross ** (annualization_factor / len(r)) - 1.0) if gross > 0 else -1.0
    # Template semantics: Sharpe = annual_return / annual_vol (safe zero-vol guard)
    sharpe = 0.0 if annual_vol <= 1e-12 else (float(annual_return) - float(risk_free_rate)) / float(annual_vol)
    if e.size == 0:
        max_drawdown = 0.0
    else:
        running_max = np.maximum.accumulate(e)
        drawdown = e / np.where(running_max == 0.0, 1.0, running_max) - 1.0
        max_drawdown = float(np.min(drawdown))
    annualized_turnover = float(np.mean(to) * annualization_factor) if to.size else 0.0
    return {
        "sharpe": float(sharpe),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "max_drawdown": float(max_drawdown),
        "annualized_turnover": float(annualized_turnover),
        "turnover_proxy": float(annualized_turnover),
        "periods": int(len(r)),
    }
