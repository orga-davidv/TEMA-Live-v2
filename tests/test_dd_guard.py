import numpy as np
import pandas as pd
from tema.risk import compute_drawdown_series, compute_dd_guard_scalar, apply_scalar_to_weights


def test_compute_drawdown_series_simple():
    eq = pd.Series([1.0, 1.2, 1.1, 1.3, 1.0])
    dd = compute_drawdown_series(eq)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0
    # after peak 1.2, 1.1 is drawdown -0.083333...
    assert round(float(dd.iloc[2]), 6) == round(1.1 / 1.2 - 1.0, 6)


def test_dd_guard_scalar_breach_and_recover():
    # synthetic drawdown: starts at 0, then breaches -0.2, stays, then recovers to 0
    idx = pd.date_range("2020-01-01", periods=6)
    dd = pd.Series([0.0, -0.05, -0.2, -0.15, -0.05, 0.0], index=idx)
    scalars = compute_dd_guard_scalar(dd, max_dd=0.1, floor=0.4, recovery_halflife=2)
    # when drawdown > -max_dd (i.e., -0.05) scalar should be near 1
    assert scalars.iloc[0] > 0.99
    assert scalars.iloc[1] > 0.9
    # at breach at index 2, scalar should move toward floor (<1)
    assert scalars.iloc[2] <= scalars.iloc[1]
    assert scalars.iloc[2] >= 0.4
    # after recovery to 0.0 at end, scalar should trend back toward 1
    assert scalars.iloc[-1] > scalars.iloc[3]
    assert scalars.iloc[-1] <= 1.0


def test_dd_guard_scalar_can_fully_derisk_below_floor_when_enabled():
    idx = pd.date_range("2020-01-01", periods=5)
    dd = pd.Series([0.0, -0.2, -0.2, -0.2, -0.2], index=idx)
    scalars = compute_dd_guard_scalar(
        dd,
        max_dd=0.1,
        floor=0.4,
        recovery_halflife=1,
        allow_full_derisk=True,
    )
    assert float(np.min(scalars.to_numpy(dtype=float))) < 0.4


def test_apply_scalar_to_weights_shape_and_values():
    weights = np.array([[0.5, 0.5], [0.6, 0.4], [1.0, 0.0]])
    scalar = np.array([1.0, 0.5, 0.2])
    adjusted = apply_scalar_to_weights(weights, scalar)
    assert adjusted.shape == weights.shape
    assert np.allclose(adjusted[0], weights[0])
    assert np.allclose(adjusted[1], weights[1] * 0.5)
    assert np.allclose(adjusted[2], weights[2] * 0.2)
