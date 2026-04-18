from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .signals.tema import generate_crossover_signal_matrix


def _ema_series(close: pd.Series, period: int) -> pd.Series:
    if int(period) <= 0:
        raise ValueError("EMA period must be > 0")
    return pd.to_numeric(close, errors="coerce").ewm(span=int(period), adjust=False).mean()


def _crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def generate_triple_ema_entry_exit_signals(
    close: pd.Series,
    combo: Tuple[int, int, int],
    *,
    shift_by: int = 1,
    logic_mode: str = "hierarchical",
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate long-only entry/exit events from a triple-EMA combo.

    logic_mode semantics:
      - "hierarchical" (default): require strict EMA stack transitions
          entry: (ema1 > ema2 > ema3) transitions from False -> True
          exit : (ema1 < ema2 < ema3) transitions from False -> True
      - "or": legacy Template-like OR crossover behavior
          entry: crossed_above(s1,s2) OR crossed_above(s1,s3) OR crossed_above(s2,s3)
          exit : crossed_below(s1,s2) OR crossed_below(s1,s3) OR crossed_below(s2,s3)

    Both entry/exit streams are shifted by `shift_by` bars before execution.
    """
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if len(combo) != 3:
        raise ValueError("combo must contain exactly three EMA periods")
    if shift_by < 0:
        raise ValueError("shift_by must be >= 0")
    if logic_mode not in {"hierarchical", "or"}:
        raise ValueError("logic_mode must be one of {'hierarchical', 'or'}")

    e1, e2, e3 = (int(combo[0]), int(combo[1]), int(combo[2]))
    s1 = _ema_series(close, e1)
    s2 = _ema_series(close, e2)
    s3 = _ema_series(close, e3)

    if logic_mode == "or":
        entries_raw = _crossed_above(s1, s2) | _crossed_above(s1, s3) | _crossed_above(s2, s3)
        exits_raw = _crossed_below(s1, s2) | _crossed_below(s1, s3) | _crossed_below(s2, s3)
    else:
        bullish_stack = (s1 > s2) & (s2 > s3)
        bearish_stack = (s1 < s2) & (s2 < s3)
        entries_raw = bullish_stack & (~bullish_stack.shift(1, fill_value=False))
        exits_raw = bearish_stack & (~bearish_stack.shift(1, fill_value=False))

    entries = entries_raw.shift(int(shift_by), fill_value=False).astype(bool)
    exits = exits_raw.shift(int(shift_by), fill_value=False).astype(bool)
    return entries, exits


def simulate_long_only_strategy_returns(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> pd.Series:
    """
    Simulate long-only strategy returns with state position in {0,1}.

    The loop mirrors template behavior:
    - position from previous bar earns current bar pct_change
    - then exits/entries toggle position (entry wins if both true)
    - turnover cost is charged when position changes.
    """
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if not isinstance(entries, pd.Series) or not isinstance(exits, pd.Series):
        raise ValueError("entries and exits must be pandas Series")

    close_num = pd.to_numeric(close, errors="coerce")
    pct = close_num.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    entries_bool = entries.reindex(close_num.index).fillna(False).astype(bool)
    exits_bool = exits.reindex(close_num.index).fillna(False).astype(bool)

    out = pd.Series(0.0, index=close_num.index, dtype=float)
    cost_rate = float(fee_rate + slippage_rate)

    pos = 0.0
    for t in range(1, len(close_num)):
        out.iloc[t] = pos * float(pct.iloc[t])

        new_pos = pos
        if bool(exits_bool.iloc[t]):
            new_pos = 0.0
        if bool(entries_bool.iloc[t]):
            new_pos = 1.0

        turnover = abs(new_pos - pos)
        out.iloc[t] -= turnover * cost_rate
        pos = new_pos

    return out


def compute_annualized_sharpe(
    returns: pd.Series,
    *,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")
    arr = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    annual_vol = float(np.std(arr, ddof=0) * np.sqrt(float(annualization)))
    if annual_vol <= 0.0 or not np.isfinite(annual_vol):
        return 0.0
    gross = float(np.prod(1.0 + arr))
    annual_return = float(gross ** (float(annualization) / arr.size) - 1.0) if gross > 0 else -1.0
    return float((annual_return - float(risk_free_rate)) / annual_vol)


def evaluate_triple_ema_combo(
    close: pd.Series,
    combo: Tuple[int, int, int],
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    shift_by: int = 1,
    annualization: float = 252.0,
    signal_logic_mode: str = "hierarchical",
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    entries, exits = generate_triple_ema_entry_exit_signals(
        close,
        combo,
        shift_by=shift_by,
        logic_mode=signal_logic_mode,
    )
    strategy_returns = simulate_long_only_strategy_returns(
        close,
        entries,
        exits,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    sharpe = compute_annualized_sharpe(
        strategy_returns,
        annualization=annualization,
        risk_free_rate=risk_free_rate,
    )
    return {
        "combo": (int(combo[0]), int(combo[1]), int(combo[2])),
        "returns": strategy_returns,
        "sharpe": float(sharpe),
    }


def select_best_triple_ema_combo(
    subtrain_close: pd.Series,
    validation_close: pd.Series,
    combos: Sequence[Tuple[int, int, int]],
    *,
    validation_shortlist: Optional[int] = 50,
    overfit_penalty: float = 0.5,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    shift_by: int = 1,
    annualization: float = 252.0,
    require_strict_order: bool = False,
    min_gap: int = 0,
    signal_logic_mode: str = "hierarchical",
    risk_free_rate: float = 0.0,
) -> tuple[Tuple[int, int, int], dict[str, Any]]:
    """
    Pick combo using validation Sharpe penalized by train/validation Sharpe gap:
      score = val_sharpe - overfit_penalty * abs(subtrain_sharpe - val_sharpe)
    """
    if not combos:
        raise ValueError("combos must not be empty")

    min_gap_i = max(0, int(min_gap))
    valid_combos: list[Tuple[int, int, int]] = []
    for combo in combos:
        c = (int(combo[0]), int(combo[1]), int(combo[2]))
        if bool(require_strict_order) and not (c[0] < c[1] < c[2]):
            continue
        if min_gap_i > 0 and ((c[1] - c[0]) < min_gap_i or (c[2] - c[1]) < min_gap_i):
            continue
        valid_combos.append(c)
    if not valid_combos:
        raise ValueError("No valid EMA combos after strict-order/min-gap constraints")

    train_rows: list[dict[str, float]] = []
    for combo in valid_combos:
        r = evaluate_triple_ema_combo(
            subtrain_close,
            combo,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            shift_by=shift_by,
            annualization=annualization,
            signal_logic_mode=signal_logic_mode,
            risk_free_rate=risk_free_rate,
        )
        train_rows.append(
            {
                "ema1_period": float(combo[0]),
                "ema2_period": float(combo[1]),
                "ema3_period": float(combo[2]),
                "subtrain_sharpe": float(r["sharpe"]),
            }
        )

    train_df = pd.DataFrame(train_rows)
    rank_col = train_df["subtrain_sharpe"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    train_df = train_df.assign(_rank=rank_col).sort_values("_rank", ascending=False).drop(columns=["_rank"])

    shortlist_n = len(train_df) if validation_shortlist is None else max(1, min(int(validation_shortlist), len(train_df)))
    shortlist = train_df.head(shortlist_n).copy()

    val_rows: list[dict[str, float]] = []
    for _, row in shortlist.iterrows():
        combo = (int(row["ema1_period"]), int(row["ema2_period"]), int(row["ema3_period"]))
        r = evaluate_triple_ema_combo(
            validation_close,
            combo,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            shift_by=shift_by,
            annualization=annualization,
            signal_logic_mode=signal_logic_mode,
            risk_free_rate=risk_free_rate,
        )
        val_rows.append(
            {
                "ema1_period": float(combo[0]),
                "ema2_period": float(combo[1]),
                "ema3_period": float(combo[2]),
                "val_sharpe": float(r["sharpe"]),
            }
        )
    val_df = pd.DataFrame(val_rows)

    merged = shortlist.merge(val_df, on=["ema1_period", "ema2_period", "ema3_period"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping results between subtrain shortlist and validation")

    gap = (merged["subtrain_sharpe"] - merged["val_sharpe"]).abs()
    merged["selection_score"] = merged["val_sharpe"] - float(overfit_penalty) * gap
    rank_sel = merged["selection_score"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    rank_val = merged["val_sharpe"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    merged = merged.assign(_rank_sel=rank_sel, _rank_val=rank_val).sort_values(
        ["_rank_sel", "_rank_val"], ascending=[False, False]
    )

    best = merged.iloc[0]
    best_combo = (int(best["ema1_period"]), int(best["ema2_period"]), int(best["ema3_period"]))
    info = {
        "subtrain_sharpe": float(best["subtrain_sharpe"]),
        "val_sharpe": float(best["val_sharpe"]),
        "selection_score": float(best["selection_score"]),
        "shortlist_size": int(shortlist_n),
        "ranking": merged.drop(columns=["_rank_sel", "_rank_val"]).reset_index(drop=True),
    }
    return best_combo, info


def build_strategy_returns_for_triple_ema_combo(
    close: pd.Series,
    combo: Tuple[int, int, int],
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    shift_by: int = 1,
    signal_logic_mode: str = "hierarchical",
) -> pd.Series:
    entries, exits = generate_triple_ema_entry_exit_signals(
        close,
        combo,
        shift_by=shift_by,
        logic_mode=signal_logic_mode,
    )
    return simulate_long_only_strategy_returns(
        close,
        entries,
        exits,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def build_train_test_strategy_returns_by_asset(
    train_close_df: pd.DataFrame,
    test_close_df: pd.DataFrame,
    combos: Sequence[Tuple[int, int, int]],
    *,
    validation_ratio: float = 0.25,
    validation_min_rows: int = 20,
    validation_shortlist: Optional[int] = 50,
    overfit_penalty: float = 0.5,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    shift_by: int = 1,
    annualization: float = 252.0,
    require_strict_order: bool = False,
    min_gap: int = 0,
    signal_logic_mode: str = "hierarchical",
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each asset, choose best triple-EMA combo on subtrain/validation and build train/test strategy returns.
    """
    if not isinstance(train_close_df, pd.DataFrame) or not isinstance(test_close_df, pd.DataFrame):
        raise ValueError("train_close_df and test_close_df must be pandas DataFrames")
    if list(train_close_df.columns) != list(test_close_df.columns):
        raise ValueError("train_close_df and test_close_df columns must match")
    if not combos:
        raise ValueError("combos must not be empty")

    train_out = pd.DataFrame(index=train_close_df.index, columns=train_close_df.columns, dtype=float)
    test_out = pd.DataFrame(index=test_close_df.index, columns=test_close_df.columns, dtype=float)
    selections: list[dict[str, Any]] = []

    for col in train_close_df.columns:
        train_close = pd.to_numeric(train_close_df[col], errors="coerce").astype(float)
        test_close = pd.to_numeric(test_close_df[col], errors="coerce").astype(float)
        n = len(train_close)
        if n < 3:
            raise ValueError(f"Asset '{col}' has insufficient train rows ({n})")

        split_idx = int(n * (1.0 - float(validation_ratio)))
        min_rows = max(1, int(validation_min_rows))
        if n > 2 * min_rows:
            split_idx = max(min_rows, min(split_idx, n - min_rows))
        split_idx = max(1, min(split_idx, n - 1))

        subtrain_close = train_close.iloc[:split_idx].copy()
        validation_close = train_close.iloc[split_idx:].copy()

        best_combo, info = select_best_triple_ema_combo(
            subtrain_close=subtrain_close,
            validation_close=validation_close,
            combos=combos,
            validation_shortlist=validation_shortlist,
            overfit_penalty=overfit_penalty,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            shift_by=shift_by,
            annualization=annualization,
            require_strict_order=require_strict_order,
            min_gap=min_gap,
            signal_logic_mode=signal_logic_mode,
            risk_free_rate=risk_free_rate,
        )

        train_out[col] = build_strategy_returns_for_triple_ema_combo(
            train_close,
            best_combo,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            shift_by=shift_by,
            signal_logic_mode=signal_logic_mode,
        ).astype(float)
        test_out[col] = build_strategy_returns_for_triple_ema_combo(
            test_close,
            best_combo,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            shift_by=shift_by,
            signal_logic_mode=signal_logic_mode,
        ).astype(float)

        selections.append(
            {
                "asset": str(col),
                "ema1_period": int(best_combo[0]),
                "ema2_period": int(best_combo[1]),
                "ema3_period": int(best_combo[2]),
                "split_idx": int(split_idx),
                "subtrain_sharpe": float(info["subtrain_sharpe"]),
                "val_sharpe": float(info["val_sharpe"]),
                "selection_score": float(info["selection_score"]),
            }
        )

    selection_df = pd.DataFrame(selections)
    return train_out, test_out, selection_df


def build_strategy_returns(
    price_df: pd.DataFrame,
    signal_df: Optional[pd.DataFrame] = None,
    *,
    # if signals not provided, build them from price series with these params
    fast_period: int = 5,
    slow_period: int = 20,
    method: str = "ema",
    shift_by: int = 1,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build per-asset strategy returns from price series and signals.

    Semantics:
    - If `signal_df` is None, signals are generated via generate_crossover_signal_matrix
      (which already applies the requested `shift_by`).
    - Strategy return for asset i at time t is:
        pos_t * pct_change_t - turnover_t * (fee_rate + slippage_rate)
      where pos_t is the signal (typically -1, 0, 1) at time t and
      turnover_t = abs(pos_t - pos_{t-1}). The initial previous position is treated as 0.
    - Returns DataFrame aligns with price_df index/columns. NaNs in prices result in NaNs in pct
      which are treated as 0.0 returns for safety.

    Args:
        price_df: DataFrame of prices (index=time, columns=assets)
        signal_df: Optional DataFrame of signals matching price_df shape. If provided, it must
            have the same columns as price_df.
        fast_period, slow_period, method, shift_by: forwarded to signal generator when
            signal_df is None.
        fee_rate, slippage_rate: cost rates applied per unit turnover (decimal, e.g. 0.001).

    Returns:
        DataFrame of per-asset periodic strategy returns (floats).
    """
    if not isinstance(price_df, pd.DataFrame):
        raise ValueError("price_df must be a pandas DataFrame")
    if price_df.empty:
        # preserve empty structure
        return pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)

    # Ensure numeric prices
    prices = price_df.apply(pd.to_numeric, errors="coerce")

    if signal_df is None:
        signal_df = generate_crossover_signal_matrix(prices, fast_period=fast_period, slow_period=slow_period, method=method, shift_by=shift_by)
    else:
        if not isinstance(signal_df, pd.DataFrame):
            raise ValueError("signal_df must be a pandas DataFrame if provided")
        if list(signal_df.columns) != list(prices.columns):
            raise ValueError("signal_df columns must match price_df columns")
        # don't modify provided signal_df (assume user handled shifting semantics); coerce numeric
        signal_df = signal_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # compute simple pct change returns, treat NaN pct as 0.0
    pct = prices.pct_change().fillna(0.0)

    total_cost_rate = float(fee_rate + slippage_rate)

    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    # For each asset compute pos * return - turnover * cost_rate
    prev_positions = pd.Series(0.0, index=prices.columns, dtype=float)

    for t in range(len(prices.index)):
        idx = prices.index[t]
        pos = signal_df.iloc[t].astype(float).fillna(0.0)
        ret = pct.iloc[t].astype(float).fillna(0.0)
        turnover = (pos - prev_positions).abs()
        # strategy return per asset
        period_return = pos * ret - turnover * total_cost_rate
        out.iloc[t] = period_return
        prev_positions = pos

    return out
