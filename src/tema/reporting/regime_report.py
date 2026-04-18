from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from tema.backtest import compute_backtest_metrics


@dataclass(frozen=True)
class RegimeBinSpec:
    bins: Sequence[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    labels: Sequence[str] = ("bear", "chop", "bull")


def _to_series(x: pd.Series | Iterable[float], index: pd.Index | None = None, name: str | None = None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(list(x), index=index)
    if name is not None:
        s.name = name
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compute_regime_report(
    *,
    returns: pd.Series,
    regime_prob: pd.Series,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0,
    spec: RegimeBinSpec = RegimeBinSpec(),
) -> pd.DataFrame:
    """Compute performance metrics by regime buckets.

    The regime is defined by a probability in [0,1] (e.g., HMM bull-state prob).
    We bucket it with pd.cut and compute Sharpe/return/vol/max-DD per bucket.
    """

    r = _to_series(returns, name="returns").fillna(0.0)
    p = _to_series(regime_prob, index=r.index, name="regime_prob").fillna(0.0)

    # Align
    joined = pd.concat([r, p], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "regime",
                "periods",
                "fraction",
                "sharpe",
                "annual_return",
                "annual_vol",
                "max_drawdown",
                "equity_final",
                "total_return",
            ]
        )

    buckets = pd.cut(
        joined["regime_prob"].clip(lower=0.0, upper=1.0),
        bins=list(spec.bins),
        labels=list(spec.labels),
        include_lowest=True,
        right=True,
    )
    joined = joined.assign(regime=buckets.astype(str))

    total_n = int(len(joined))
    rows: list[dict] = []
    for regime, sub in joined.groupby("regime", dropna=False):
        rr = sub["returns"].to_numpy(dtype=float)
        eq = np.cumprod(1.0 + rr) if rr.size else np.array([])
        m = compute_backtest_metrics(
            rr,
            eq,
            np.zeros_like(rr),
            float(annualization_factor),
            risk_free_rate=float(risk_free_rate),
        )
        rows.append(
            {
                "regime": str(regime),
                "periods": int(m.get("periods", len(rr))),
                "fraction": float(len(rr) / max(1, total_n)),
                "sharpe": float(m.get("sharpe", np.nan)),
                "annual_return": float(m.get("annual_return", np.nan)),
                "annual_vol": float(m.get("annual_vol", np.nan)),
                "max_drawdown": float(m.get("max_drawdown", np.nan)),
                "equity_final": float(eq[-1]) if eq.size else 1.0,
                "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    # stable order
    order = {lbl: i for i, lbl in enumerate(spec.labels)}
    df["_order"] = df["regime"].map(order).fillna(9999)
    df = df.sort_values(["_order"]).drop(columns=["_order"]).reset_index(drop=True)
    return df
