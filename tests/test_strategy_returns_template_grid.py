import numpy as np
import pandas as pd

from tema.strategy_returns import (
    build_train_test_strategy_returns_by_asset,
    compute_annualized_sharpe,
    generate_triple_ema_entry_exit_signals,
    select_best_triple_ema_combo,
    simulate_long_only_strategy_returns,
)
from tema.backtest import compute_backtest_metrics


def test_generate_triple_ema_entry_exit_signals_defaults_to_hierarchical_stack_transitions():
    close = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 99.0, 101.0], index=range(7))
    entries, exits = generate_triple_ema_entry_exit_signals(close, (2, 3, 5), shift_by=1)

    s1 = close.ewm(span=2, adjust=False).mean()
    s2 = close.ewm(span=3, adjust=False).mean()
    s3 = close.ewm(span=5, adjust=False).mean()
    bullish_stack = (s1 > s2) & (s2 > s3)
    bearish_stack = (s1 < s2) & (s2 < s3)

    expected_entries = (bullish_stack & (~bullish_stack.shift(1, fill_value=False))).shift(1, fill_value=False)
    expected_exits = (bearish_stack & (~bearish_stack.shift(1, fill_value=False))).shift(1, fill_value=False)

    assert entries.equals(expected_entries.astype(bool))
    assert exits.equals(expected_exits.astype(bool))


def test_generate_triple_ema_entry_exit_signals_supports_legacy_or_mode():
    close = pd.Series([100.0, 101.0, 103.0, 102.0, 100.0, 99.0, 101.0], index=range(7))
    entries, exits = generate_triple_ema_entry_exit_signals(close, (2, 3, 5), shift_by=1, logic_mode="or")

    s1 = close.ewm(span=2, adjust=False).mean()
    s2 = close.ewm(span=3, adjust=False).mean()
    s3 = close.ewm(span=5, adjust=False).mean()
    crossed_above = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))
    crossed_below = lambda a, b: (a < b) & (a.shift(1) >= b.shift(1))
    expected_entries = (crossed_above(s1, s2) | crossed_above(s1, s3) | crossed_above(s2, s3)).shift(1, fill_value=False)
    expected_exits = (crossed_below(s1, s2) | crossed_below(s1, s3) | crossed_below(s2, s3)).shift(1, fill_value=False)

    assert entries.equals(expected_entries.astype(bool))
    assert exits.equals(expected_exits.astype(bool))


def test_simulate_long_only_strategy_returns_toggle_and_costs():
    close = pd.Series([100.0, 110.0, 121.0, 110.0], index=range(4))
    entries = pd.Series([False, True, False, False], index=close.index)
    exits = pd.Series([False, False, False, True], index=close.index)

    out = simulate_long_only_strategy_returns(close, entries, exits, fee_rate=0.01, slippage_rate=0.0)

    expected = np.array(
        [
            0.0,  # template loop starts from t=1
            -0.01,  # enter at t=1: turnover=1 cost charged
            0.1,  # remain long through t=2
            (110.0 / 121.0 - 1.0) - 0.01,  # exit at t=3: long return + cost
        ]
    )
    assert np.allclose(out.to_numpy(dtype=float), expected, atol=1e-12)


def test_select_best_triple_ema_combo_uses_validation_gap_penalty(monkeypatch):
    subtrain = pd.Series([1.0, 2.0, 3.0], name="subtrain")
    validation = pd.Series([1.0, 1.5, 2.0], name="validation")
    combos = [(2, 5, 8), (3, 6, 9)]

    def _fake_eval(close, combo, **kwargs):
        if close.name == "subtrain":
            sharpe = 2.0 if combo == (2, 5, 8) else 1.6
        else:
            sharpe = 1.0 if combo == (2, 5, 8) else 1.5
        return {"combo": combo, "returns": pd.Series(dtype=float), "sharpe": sharpe}

    monkeypatch.setattr("tema.strategy_returns.evaluate_triple_ema_combo", _fake_eval)

    best_combo, info = select_best_triple_ema_combo(
        subtrain,
        validation,
        combos,
        validation_shortlist=2,
        overfit_penalty=0.5,
    )

    assert best_combo == (3, 6, 9)
    assert np.isclose(info["subtrain_sharpe"], 1.6)
    assert np.isclose(info["val_sharpe"], 1.5)
    assert np.isclose(info["selection_score"], 1.5 - 0.5 * abs(1.6 - 1.5))


def test_select_best_triple_ema_combo_enforces_min_gap_and_order(monkeypatch):
    subtrain = pd.Series([1.0, 2.0, 3.0], name="subtrain")
    validation = pd.Series([1.0, 1.5, 2.0], name="validation")
    combos = [(2, 3, 4), (2, 5, 8), (8, 5, 2)]

    def _fake_eval(close, combo, **kwargs):
        return {"combo": combo, "returns": pd.Series(dtype=float), "sharpe": 1.0}

    monkeypatch.setattr("tema.strategy_returns.evaluate_triple_ema_combo", _fake_eval)
    best_combo, _ = select_best_triple_ema_combo(
        subtrain,
        validation,
        combos,
        validation_shortlist=2,
        require_strict_order=True,
        min_gap=2,
    )
    assert best_combo == (2, 5, 8)


def test_compute_annualized_sharpe_matches_backtest_metric_semantics_with_rf():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01], dtype=float)
    rf = 0.02
    sharpe = compute_annualized_sharpe(returns, annualization=252.0, risk_free_rate=rf)
    eq = np.cumprod(1.0 + returns.to_numpy(dtype=float))
    metrics = compute_backtest_metrics(
        returns.to_numpy(dtype=float),
        eq,
        np.zeros(len(returns), dtype=float),
        252.0,
        risk_free_rate=rf,
    )
    assert np.isclose(sharpe, metrics["sharpe"])


def test_compute_annualized_sharpe_zero_vol_matches_backtest_zero_guard():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0], dtype=float)
    sharpe = compute_annualized_sharpe(returns, annualization=252.0, risk_free_rate=0.01)
    eq = np.cumprod(1.0 + returns.to_numpy(dtype=float))
    metrics = compute_backtest_metrics(
        returns.to_numpy(dtype=float),
        eq,
        np.zeros(len(returns), dtype=float),
        252.0,
        risk_free_rate=0.01,
    )
    assert sharpe == 0.0
    assert sharpe == metrics["sharpe"]


def test_build_train_test_strategy_returns_by_asset_uses_selected_combo(monkeypatch):
    train_df = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "B": [200.0, 198.0, 197.0, 199.0, 201.0, 202.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "A": [106.0, 107.0, 108.0],
            "B": [203.0, 204.0, 205.0],
        }
    )
    combos = [(2, 4, 8), (3, 6, 9)]

    def _fake_select(subtrain_close, validation_close, combos, **kwargs):
        if subtrain_close.name == "A":
            return (2, 4, 8), {"subtrain_sharpe": 1.0, "val_sharpe": 0.8, "selection_score": 0.7}
        return (3, 6, 9), {"subtrain_sharpe": 0.9, "val_sharpe": 1.1, "selection_score": 1.0}

    def _fake_build(close, combo, **kwargs):
        return pd.Series(np.full(len(close), float(combo[0]) / 100.0), index=close.index, dtype=float)

    monkeypatch.setattr("tema.strategy_returns.select_best_triple_ema_combo", _fake_select)
    monkeypatch.setattr("tema.strategy_returns.build_strategy_returns_for_triple_ema_combo", _fake_build)

    train_ret, test_ret, selection = build_train_test_strategy_returns_by_asset(
        train_df,
        test_df,
        combos,
        validation_ratio=0.25,
        validation_min_rows=2,
    )

    assert list(train_ret.columns) == ["A", "B"]
    assert list(test_ret.columns) == ["A", "B"]
    assert np.allclose(train_ret["A"].to_numpy(), 0.02)
    assert np.allclose(train_ret["B"].to_numpy(), 0.03)
    assert np.allclose(test_ret["A"].to_numpy(), 0.02)
    assert np.allclose(test_ret["B"].to_numpy(), 0.03)
    assert set(selection["asset"]) == {"A", "B"}
