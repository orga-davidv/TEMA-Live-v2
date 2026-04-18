from tema import BacktestConfig, Runner
from tema.turnover import apply_rebalance_gating_with_diagnostics


def test_should_respect_min_threshold():
    cfg = BacktestConfig(rebalance_min_threshold=0.01, cost_aware_rebalance=False)
    r = Runner(cfg)
    current = [0.0]
    candidate = [0.005]
    expected_alphas = [0.1]

    out = r.decide_portfolio_weights(current, candidate, expected_alphas)
    assert out[0] == 0.0  # below min threshold -> no rebalance


def test_cost_aware_gate_blocks_when_alpha_low():
    cfg = BacktestConfig(rebalance_min_threshold=0.0001, cost_aware_rebalance=True, cost_aware_rebalance_multiplier=1.0, fee_rate=0.001, slippage_rate=0.001)
    r = Runner(cfg)
    current = [0.0]
    candidate = [0.5]
    # expected alpha small compared to implied annualized costs
    expected_alphas = [0.0001]

    out = r.decide_portfolio_weights(current, candidate, expected_alphas)
    assert out[0] == 0.0  # gated off because expected alpha < costs threshold


def test_cost_aware_gate_allows_when_alpha_high():
    cfg = BacktestConfig(rebalance_min_threshold=0.0001, cost_aware_rebalance=True, cost_aware_rebalance_multiplier=1.0, fee_rate=0.0005, slippage_rate=0.0005)
    r = Runner(cfg)
    current = [0.0]
    candidate = [0.5]
    # expected alpha large enough to clear costs
    expected_alphas = [1.0]

    out = r.decide_portfolio_weights(current, candidate, expected_alphas)
    assert out[0] == 0.5


def test_ml_and_vol_flags_present():
    cfg = BacktestConfig()
    r = Runner(cfg)
    flags = r.ml_and_vol_flags()
    assert "ml_enabled" in flags and "vol_target_enabled" in flags


def test_rebalance_gate_diagnostics_include_cost_blocks():
    cfg = BacktestConfig(
        rebalance_min_threshold=0.0001,
        cost_aware_rebalance=True,
        cost_aware_rebalance_multiplier=1.0,
        fee_rate=0.001,
        slippage_rate=0.001,
    )
    gated, diag = apply_rebalance_gating_with_diagnostics(
        current_weights=[0.0, 0.0],
        candidate_weights=[0.5, 0.005],
        expected_alphas=[0.0001, 0.1],
        cfg=cfg,
    )
    assert gated == [0.0, 0.005]
    assert diag["cost_block_count"] == 1
    assert diag["threshold_block_count"] == 0
