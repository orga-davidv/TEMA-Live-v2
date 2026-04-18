import numpy as np
import pandas as pd

from tema.config import BacktestConfig
from tema.pipeline import runner as pipeline_runner


def test_build_template_grid_combos_is_deterministic_and_filters_invalid():
    cfg = BacktestConfig(
        template_default_universe=True,
        template_grid_short_periods=(3, 5, 3),
        template_grid_mid_periods=(4, 8),
        template_grid_long_periods=(8, 10),
        template_grid_require_strict_order=True,
    )
    combos = pipeline_runner._build_template_grid_combos(cfg)
    assert combos == [(3, 4, 8), (3, 4, 10), (3, 8, 10), (5, 8, 10)]


def test_build_template_grid_combos_enforces_min_gap_when_configured():
    cfg = BacktestConfig(
        template_default_universe=True,
        template_grid_short_periods=(3, 4),
        template_grid_mid_periods=(5, 7),
        template_grid_long_periods=(8, 10),
        template_grid_require_strict_order=True,
        template_grid_min_gap=3,
    )
    combos = pipeline_runner._build_template_grid_combos(cfg)
    assert combos == [(3, 7, 10), (4, 7, 10)]


def test_load_data_context_template_uses_train_test_grid_builder(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    price_df = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105], "B": [50, 50, 51, 52, 53, 54]}, index=idx, dtype=float)
    train_df = price_df.iloc[:4].copy()
    test_df = price_df.iloc[4:].copy()

    calls = {}

    def _fake_grid_builder(train_close_df, test_close_df, combos, **kwargs):
        calls["train_shape"] = train_close_df.shape
        calls["test_shape"] = test_close_df.shape
        calls["combos"] = list(combos)
        calls["kwargs"] = dict(kwargs)
        train_ret = pd.DataFrame(0.001, index=train_close_df.index, columns=train_close_df.columns)
        test_ret = pd.DataFrame(0.002, index=test_close_df.index, columns=test_close_df.columns)
        selection = pd.DataFrame(
            [
                {"asset": "A", "ema1_period": 3, "ema2_period": 8, "ema3_period": 21, "split_idx": 3, "subtrain_sharpe": 1.0, "val_sharpe": 0.8, "selection_score": 0.7},
                {"asset": "B", "ema1_period": 5, "ema2_period": 13, "ema3_period": 34, "split_idx": 3, "subtrain_sharpe": 0.9, "val_sharpe": 0.7, "selection_score": 0.6},
            ]
        )
        return train_ret, test_ret, selection

    monkeypatch.setattr("tema.pipeline.runner.load_price_panel", lambda **kwargs: price_df)
    monkeypatch.setattr("tema.pipeline.runner.split_panel_per_asset", lambda *args, **kwargs: (train_df, test_df))
    monkeypatch.setattr("tema.pipeline.runner.build_train_test_strategy_returns_by_asset", _fake_grid_builder)

    cfg = BacktestConfig(modular_data_signals_enabled=True, template_default_universe=True)
    ctx = pipeline_runner._load_data_context(cfg)

    assert calls["train_shape"] == train_df.shape
    assert calls["test_shape"] == test_df.shape
    assert len(calls["combos"]) > 0
    assert calls["kwargs"]["require_strict_order"] is True
    assert calls["kwargs"]["min_gap"] == 0
    assert calls["kwargs"]["signal_logic_mode"] == "hierarchical"
    assert ctx["strategy_grid_diagnostics"]["mode"] == "template_train_validation_grid"
    assert ctx["strategy_grid_diagnostics"]["selected_assets"] == 2
    assert len(ctx["strategy_combo_selection"]) == 2
    assert np.allclose(ctx["train_strategy_returns"].to_numpy(dtype=float), 0.001)
    assert np.allclose(ctx["test_strategy_returns"].to_numpy(dtype=float), 0.002)


def test_portfolio_stage_template_uses_geometric_annualized_expected_alphas(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    train_df = pd.DataFrame({"A": [100, 101, 102, 103], "B": [100, 100, 100, 100]}, index=idx, dtype=float)
    test_df = pd.DataFrame({"A": [104, 105], "B": [100, 100]}, index=pd.date_range("2024-01-05", periods=2, freq="D"), dtype=float)
    train_returns = train_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_strategy_returns = pd.DataFrame({"A": [0.001, 0.002, -0.001, 0.0015], "B": [0.0, 0.0, 0.0, 0.0]}, index=idx)

    class StubEngine:
        def generate(self, price_df, fast_period, slow_period, method):
            return pd.DataFrame({"A": [1.0] * len(price_df), "B": [1.0] * len(price_df)}, index=price_df.index)

    monkeypatch.setattr("tema.pipeline.runner.resolve_signal_engine", lambda use_cpp, cpp_engine=None: StubEngine())

    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        template_default_universe=True,
        portfolio_modular_enabled=False,
        freq="D",
    )
    ctx = {
        "price_df": pd.concat([train_df, test_df]),
        "train_df": train_df,
        "test_df": test_df,
        "train_returns": train_returns,
        "train_strategy_returns": train_strategy_returns,
        "strategy_returns_include_costs": True,
        "split_mode": "per_asset",
        "max_assets_used": None,
        "full_universe_override": True,
        "min_rows_used": 400,
        "train_ratio_used": 0.6,
        "strategy_combo_selection": [{"asset": "A", "ema1_period": 3, "ema2_period": 8, "ema3_period": 21}],
        "strategy_grid_diagnostics": {"mode": "template_train_validation_grid"},
    }

    _, _, expected_alphas, info, _ = pipeline_runner._portfolio_stage(cfg, data_context=ctx)
    expected_a = float(np.expm1(np.mean(np.log1p(np.array([0.001, 0.002, -0.001, 0.0015], dtype=float))) * 252.0))

    assert np.isclose(expected_alphas[0], expected_a)
    assert expected_alphas[1] == 0.0
    assert info["expected_alpha_source"] == "strategy_train_returns_geometric_annualized"
    assert info["expected_alpha_method"] == "geometric_annualized_per_asset"
    assert info["strategy_combo_selection"][0]["asset"] == "A"
