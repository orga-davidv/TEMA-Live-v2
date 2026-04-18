import json
from pathlib import Path

import numpy as np

from tema.backtest import build_weight_schedule_from_signals, run_return_equity_simulation
from tema.config import BacktestConfig
from tema.pipeline import run_pipeline


def test_run_return_equity_simulation_computes_core_metrics():
    asset_returns = np.array([[0.01], [-0.02], [0.03]], dtype=float)
    target_weights = np.array([[1.0], [1.0], [1.0]], dtype=float)
    res = run_return_equity_simulation(asset_returns, target_weights, freq="D")

    expected_vol = float(np.std(asset_returns[:, 0], ddof=0) * np.sqrt(252.0))
    expected_equity = np.cumprod(1.0 + asset_returns[:, 0])
    expected_mdd = float(np.min(expected_equity / np.maximum.accumulate(expected_equity) - 1.0))
    gross = float(np.prod(1.0 + asset_returns[:, 0]))
    expected_annual_return = float(gross ** (252.0 / len(asset_returns[:, 0])) - 1.0) if gross > 0 else -1.0
    expected_sharpe = 0.0 if expected_vol <= 1e-12 else float(expected_annual_return) / float(expected_vol)

    assert len(res.periodic_returns) == 3
    assert abs(res.metrics["sharpe"] - expected_sharpe) < 1e-10
    assert abs(res.metrics["annual_vol"] - expected_vol) < 1e-10
    assert abs(res.metrics["max_drawdown"] - expected_mdd) < 1e-10
    assert "annualized_turnover" in res.metrics


def test_run_return_equity_simulation_applies_risk_free_rate_to_sharpe():
    asset_returns = np.array([[0.01], [-0.02], [0.03]], dtype=float)
    target_weights = np.array([[1.0], [1.0], [1.0]], dtype=float)
    rf = 0.02
    res = run_return_equity_simulation(asset_returns, target_weights, freq="D", risk_free_rate=rf)

    expected_vol = float(np.std(asset_returns[:, 0], ddof=0) * np.sqrt(252.0))
    gross = float(np.prod(1.0 + asset_returns[:, 0]))
    expected_annual_return = float(gross ** (252.0 / len(asset_returns[:, 0])) - 1.0) if gross > 0 else -1.0
    expected_sharpe = 0.0 if expected_vol <= 1e-12 else (expected_annual_return - rf) / expected_vol
    assert abs(res.metrics["sharpe"] - expected_sharpe) < 1e-10


def test_build_weight_schedule_from_signals_has_safe_fallback():
    import pandas as pd

    signal_df = pd.DataFrame([[1.0, 0.0], [0.0, 0.0], [-1.0, 2.0]], columns=["a", "b"])
    sched = build_weight_schedule_from_signals(signal_df, fallback_weights=[0.6, 0.4])
    assert sched.shape == (3, 2)
    assert abs(float(np.sum(sched[0])) - 1.0) < 1e-12
    assert abs(float(np.sum(sched[1])) - 1.0) < 1e-12
    assert abs(sched[1, 0] - 0.6) < 1e-12 and abs(sched[1, 1] - 0.4) < 1e-12


def test_pipeline_includes_performance_artifact_with_fallback(tmp_path):
    out_root = tmp_path / "outputs"
    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        data_path=str(tmp_path / "missing-data-dir"),
    )
    res = run_pipeline(run_id="perf-fallback-test", cfg=cfg, out_root=str(out_root))
    perf_path = Path(res["out_dir"]) / "performance.json"
    manifest = json.loads((Path(res["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    performance = json.loads(perf_path.read_text(encoding="utf-8"))

    assert perf_path.exists()
    assert "performance" in manifest["artifacts"]
    assert performance["fallback_used"] is True
    for key in ("sharpe", "annual_return", "annual_vol", "max_drawdown", "annualized_turnover", "turnover_proxy"):
        assert key in performance


def test_template_default_universe_enables_static_weight_schedule(monkeypatch):
    import pandas as pd
    from tema.pipeline.runner import _backtest_stage
    from tema.backtest import run_return_equity_simulation

    # Build a tiny deterministic price panel with 2 assets and 4 rows
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    price_df = pd.DataFrame(
        {
            "a": [100.0, 101.0, 102.0, 103.0],
            "b": [100.0, 100.5, 101.0, 101.5],
        },
        index=idx,
    )
    train_df = price_df.iloc[:2]
    test_df = price_df.iloc[2:]
    train_returns = train_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)

    data_context = {
        "price_df": price_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_returns": train_returns,
        "max_assets_used": 2,
        "full_universe_override": False,
        "min_rows_used": 3,
        "train_ratio_used": 0.5,
    }

    # final_weights start balanced between a and b
    final_weights = [0.5, 0.5]

    # Stub signal engine always favors asset 'b' so blended weights would differ
    class StubEngine:
        def generate(self, price_df, fast_period, slow_period, method):
            df = pd.DataFrame({"a": [0.0, 0.0], "b": [1.0, 1.0]}, index=test_df.index)
            return df

    monkeypatch.setattr("tema.pipeline.runner.resolve_signal_engine", lambda use_cpp, cpp_engine=None: StubEngine())

    from tema.config import BacktestConfig

    # Case 1: template-default-universe -> static weights should be used
    cfg_static = BacktestConfig(modular_data_signals_enabled=True, template_default_universe=True)
    cfg_static.backtest_static_weights_in_template = True
    perf_static = _backtest_stage(cfg_static, final_weights, [0.01, 0.02], data_context=data_context)

    # Compute expected equity final using constant weight schedule (no signals)
    returns_np = test_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0).to_numpy(dtype=float)
    const_weights = np.tile(np.asarray(final_weights, dtype=float), (returns_np.shape[0], 1))
    sim_const = run_return_equity_simulation(asset_returns=returns_np, target_weights=const_weights, freq="D")

    assert abs(perf_static["equity_final"] - float(sim_const.equity_curve[-1])) < 1e-12

    # Case 2: non-template mode -> signal-derived blending should alter weights
    cfg_dyn = BacktestConfig(modular_data_signals_enabled=True, template_default_universe=False)
    cfg_dyn.backtest_static_weights_in_template = False
    perf_dyn = _backtest_stage(cfg_dyn, final_weights, [0.01, 0.02], data_context=data_context)

    # Expect the dynamic case to differ from the constant case because StubEngine favors 'b'
    assert abs(perf_dyn["equity_final"] - float(sim_const.equity_curve[-1])) > 1e-6


def test_template_backtest_uses_strategy_returns_without_double_costing():
    import pandas as pd
    from tema.pipeline.runner import _backtest_stage

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    price_df = pd.DataFrame(
        {
            "a": [100.0, 101.0, 102.0, 103.0, 104.0],
            "b": [50.0, 49.0, 48.0, 49.0, 50.0],
        },
        index=idx,
    )
    train_df = price_df.iloc[:3]
    test_df = price_df.iloc[3:]
    strategy_test_returns = pd.DataFrame(
        {"a": [0.01, -0.005], "b": [0.02, 0.01]},
        index=test_df.index,
    )
    train_returns = train_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    data_context = {
        "price_df": price_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_returns": train_returns,
        "test_strategy_returns": strategy_test_returns,
        "strategy_returns_include_costs": True,
        "split_mode": "per_asset",
    }
    cfg = BacktestConfig(
        modular_data_signals_enabled=False,
        template_default_universe=True,
        fee_rate=0.02,
        slippage_rate=0.03,
    )
    final_weights = [0.5, 0.5]
    perf = _backtest_stage(cfg, final_weights, [0.01, 0.01], data_context=data_context)

    expected = run_return_equity_simulation(
        asset_returns=strategy_test_returns.to_numpy(dtype=float),
        target_weights=np.tile(np.asarray(final_weights, dtype=float), (len(strategy_test_returns), 1)),
        fee_rate=0.0,
        slippage_rate=0.0,
        freq="D",
    )
    assert abs(perf["equity_final"] - float(expected.equity_curve[-1])) < 1e-12
    assert perf["source"]["returns_source"] == "strategy_test_returns"
    assert perf["source"]["strategy_returns_include_costs"] is True
