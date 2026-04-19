import math

import numpy as np
import pytest

from tema.leverage.confluence import ConfluenceConfig, align_sign, compute_confluence_score, winsorize, zscore
from tema.leverage.engine import LeverageEngineConfig, compute_leverage
from tema.leverage.gates import (
    HardGateConfig,
    apply_correlation_alert_cap,
    apply_event_blackout_cap,
    apply_hard_gates,
    apply_liquidity_gate,
)
from tema.leverage.mapping import ConfluenceMappingConfig, map_confluence_to_multiplier


def test_normalization_utilities_deterministic_behavior():
    values = [-10.0, -1.0, 0.0, 1.0, 10.0]
    clipped = winsorize(values, lower_quantile=0.20, upper_quantile=0.80)
    assert clipped.shape == (5,)

    lo = float(np.quantile(np.asarray(values, dtype=float), 0.20))
    hi = float(np.quantile(np.asarray(values, dtype=float), 0.80))
    assert float(np.min(clipped)) >= lo
    assert float(np.max(clipped)) <= hi

    assert np.allclose(zscore([3.0, 3.0, 3.0]), np.zeros(3))
    assert np.allclose(align_sign([1.0, -2.0, 0.5], sign=-1.0), np.array([-1.0, 2.0, -0.5]))


def test_compute_confluence_score_in_unit_interval_and_repeatable():
    cfg = ConfluenceConfig(winsor_lower_quantile=0.0, winsor_upper_quantile=1.0)
    signals = {"carry": 0.3, "trend": 2.0, "risk": -0.2}
    weights = {"trend": 2.5, "carry": 0.3, "risk": 0.2}
    sign_map = {"risk": -1.0}

    score_1 = compute_confluence_score(signals, weights=weights, sign_map=sign_map, cfg=cfg)
    score_2 = compute_confluence_score(signals, weights=weights, sign_map=sign_map, cfg=cfg)

    assert 0.0 <= score_1 <= 1.0
    assert math.isclose(score_1, score_2, rel_tol=0.0, abs_tol=1e-15)
    assert score_1 > 0.5


def test_mapping_modes_and_config_validation():
    linear = ConfluenceMappingConfig(mode="linear", min_multiplier=0.5, max_multiplier=1.5)
    assert math.isclose(map_confluence_to_multiplier(0.25, linear), 0.75, rel_tol=0.0, abs_tol=1e-12)

    stepwise = ConfluenceMappingConfig(
        mode="stepwise",
        min_multiplier=0.0,
        max_multiplier=2.0,
        step_thresholds=(0.30, 0.70),
        step_multipliers=(0.50, 1.00, 1.40),
    )
    assert math.isclose(map_confluence_to_multiplier(0.10, stepwise), 0.50, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(map_confluence_to_multiplier(0.50, stepwise), 1.00, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(map_confluence_to_multiplier(0.90, stepwise), 1.40, rel_tol=0.0, abs_tol=1e-12)

    kelly = ConfluenceMappingConfig(mode="kelly_shrink", min_multiplier=0.5, max_multiplier=1.5, kelly_gamma=2.0)
    assert math.isclose(map_confluence_to_multiplier(0.50, kelly), 0.75, rel_tol=0.0, abs_tol=1e-12)

    with pytest.raises(ValueError):
        ConfluenceMappingConfig(mode="unknown")
    with pytest.raises(ValueError):
        ConfluenceMappingConfig(mode="stepwise", step_thresholds=(0.7, 0.3), step_multipliers=(0.5, 1.0, 1.5))
    with pytest.raises(ValueError):
        ConfluenceMappingConfig(mode="stepwise", step_thresholds=(0.4,), step_multipliers=(0.5,))


def test_hard_gate_primitives_apply_expected_overrides():
    assert math.isclose(apply_event_blackout_cap(1.2, event_blackout=True, cap=0.5), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(apply_event_blackout_cap(1.2, event_blackout=False, cap=0.5), 1.2, rel_tol=0.0, abs_tol=1e-12)

    assert math.isclose(
        apply_liquidity_gate(1.0, spread_z=2.5, depth_percentile=0.5, reduction_factor=0.25),
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        apply_liquidity_gate(1.0, spread_z=0.5, depth_percentile=0.05, depth_threshold=0.1, reduction_factor=0.25),
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        apply_liquidity_gate(1.0, spread_z=0.5, depth_percentile=0.5, reduction_factor=0.25),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(apply_correlation_alert_cap(1.5, correlation_alert=True, cap=1.0), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(apply_correlation_alert_cap(0.8, correlation_alert=False, cap=1.0), 0.8, rel_tol=0.0, abs_tol=1e-12)

    gated, flags = apply_hard_gates(
        leverage=2.0,
        event_blackout=True,
        spread_z=3.0,
        depth_percentile=0.05,
        correlation_alert=True,
        cfg=HardGateConfig(event_blackout_cap=0.5, liquidity_reduction_factor=0.25, correlation_alert_cap=1.0),
    )
    assert math.isclose(gated, 0.125, rel_tol=0.0, abs_tol=1e-12)
    assert flags == {"event_blackout": True, "liquidity": True, "correlation_alert": True}


def test_compute_leverage_combines_mapping_gates_and_clipping():
    engine_cfg = LeverageEngineConfig(
        mapping=ConfluenceMappingConfig(mode="linear", min_multiplier=0.5, max_multiplier=1.5),
        gates=HardGateConfig(event_blackout_cap=0.5, liquidity_reduction_factor=0.25, correlation_alert_cap=1.0),
        leverage_floor=0.10,
        leverage_cap=2.0,
    )

    value_1, diag_1 = compute_leverage(
        base_leverage=2.0,
        confluence_score=1.0,
        cfg=engine_cfg,
        event_blackout=True,
        spread_z=3.0,
        depth_percentile=0.50,
        correlation_alert=True,
        return_diagnostics=True,
    )
    value_2, diag_2 = compute_leverage(
        base_leverage=2.0,
        confluence_score=1.0,
        cfg=engine_cfg,
        event_blackout=True,
        spread_z=3.0,
        depth_percentile=0.50,
        correlation_alert=True,
        return_diagnostics=True,
    )

    assert math.isclose(value_1, 0.125, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(value_1, value_2, rel_tol=0.0, abs_tol=1e-15)
    assert diag_1 == diag_2
    assert diag_1["gate_flags"] == {"event_blackout": True, "liquidity": True, "correlation_alert": True}

    clipped = compute_leverage(base_leverage=2.0, confluence_score=1.0, cfg=engine_cfg)
    assert math.isclose(clipped, 2.0, rel_tol=0.0, abs_tol=1e-12)

    with pytest.raises(ValueError):
        compute_leverage(base_leverage=float("nan"), confluence_score=0.5, cfg=engine_cfg)
