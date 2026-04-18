import numpy as np

from tema.backtest import run_return_equity_simulation


def test_default_model_matches_manual_fee_slippage():
    asset_returns = np.array([[0.01], [-0.01], [0.02]], dtype=float)
    target_weights = np.array([[1.0], [1.0], [1.0]], dtype=float)
    # manual computation using legacy formula: turnover * (fee + slippage)
    executed = np.vstack([target_weights[0], target_weights[:-1]])
    prev = executed[0]
    manual = []
    fee = 0.0005
    slippage = 0.0005
    for t in range(asset_returns.shape[0]):
        cur = executed[t]
        turnover = float(np.sum(np.abs(cur - prev)))
        pnl = float(np.dot(cur, asset_returns[t]))
        manual.append(pnl - turnover * (fee + slippage))
        prev = cur

    sim = run_return_equity_simulation(asset_returns, target_weights, fee_rate=fee, slippage_rate=slippage, freq="D")
    assert np.allclose(np.asarray(sim.periodic_returns, dtype=float), np.asarray(manual, dtype=float))


def test_extended_model_spread_impact_borrow_no_nan_and_short_detection():
    # two periods, single asset where second period is net short
    asset_returns = np.array([[0.01], [0.02]], dtype=float)
    target_weights = np.array([[0.0], [-1.0]], dtype=float)

    sim = run_return_equity_simulation(
        asset_returns,
        target_weights,
        fee_rate=0.0,
        slippage_rate=0.0,
        cost_model="extended",
        spread_bps=1.0,
        impact_coeff=0.001,
        borrow_bps=10.0,
        freq="D",
    )
    # no NaN or inf
    assert np.all(np.isfinite(np.asarray(sim.periodic_returns, dtype=float)))

    # Borrow fee should make extended model with borrow_bps > 0 produce <= returns than without borrow
    sim_no_borrow = run_return_equity_simulation(
        asset_returns,
        target_weights,
        fee_rate=0.0,
        slippage_rate=0.0,
        cost_model="extended",
        spread_bps=1.0,
        impact_coeff=0.001,
        borrow_bps=0.0,
        freq="D",
    )
    assert any(np.array(sim.periodic_returns) <= np.array(sim_no_borrow.periodic_returns))
