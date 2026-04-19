from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestConfig:
    # Modular data/signals toggle (safe-by-default off)
    modular_data_signals_enabled: bool = False
    data_path: Optional[str] = None
    data_max_assets: int = 3
    data_full_universe_for_parity: bool = True
    data_min_rows: int = 30
    data_train_ratio: float = 0.7

    # Data quality checks (optional; reports artifact and can fail-fast)
    data_quality_enabled: bool = False
    data_quality_fail_fast: bool = False
    data_quality_max_nan_frac: float = 0.05
    data_quality_max_gap_days: float = 7.0
    data_quality_min_price: float = 1e-12

    # Template-style universe profile: merged_d1 source, min_history_rows=400, train_ratio=0.60, full asset set
    template_default_universe: bool = False
    # When True, template_default_universe will reuse precomputed Template/*.csv artifacts
    # (asset_strategy_summary.csv + black_litterman_weights.csv) to match benchmark results
    # deterministically without re-running the expensive per-asset grid search.
    template_use_precomputed_artifacts: bool = True
    # Template grid search controls (triple-EMA per-asset validation path)
    template_grid_short_periods: tuple[int, ...] = (3, 4, 5, 6)
    template_grid_mid_periods: tuple[int, ...] = (8, 13, 21, 34)
    template_grid_long_periods: tuple[int, ...] = (55, 89, 144, 233)
    template_grid_require_strict_order: bool = True
    template_grid_min_gap: int = 0
    template_grid_signal_logic: str = "hierarchical"
    template_grid_validation_ratio: float = 0.25
    template_grid_validation_min_rows: int = 20
    template_grid_validation_shortlist: Optional[int] = 50
    template_grid_overfit_penalty: float = 0.5
    template_grid_shift_by: int = 1
    # Computed template mode: keep deterministic asset universe aligned to benchmark asset list
    # while still computing combos/weights from source data (no precomputed weight override).
    template_computed_lock_benchmark_universe: bool = True
    signal_fast_period: int = 5
    signal_slow_period: int = 20
    signal_method: str = "ema"
    signal_use_cpp: bool = False
    portfolio_modular_enabled: bool = False
    portfolio_method: str = "bl"
    portfolio_use_hrp_hook: bool = False
    portfolio_use_nco_hook: bool = False
    portfolio_cov_shrinkage: float = 0.15
    portfolio_bl_tau: float = 0.05
    portfolio_bl_view_confidence: float = 0.65
    portfolio_bl_omega_scale: float = 0.25
    portfolio_bl_max_weight: float = 0.15
    portfolio_risk_aversion: float = 2.5
    portfolio_min_weight: float = 0.0
    portfolio_max_weight: float = 1.0

    # Turnover / rebalance controls (Phase 2b)
    rebalance_min_threshold: float = 0.001
    cost_aware_rebalance: bool = False
    cost_aware_rebalance_multiplier: float = 1.0
    cost_aware_alpha_lookback: int = 20

    # Penalty applied during selection/optimization: Sharpe - lambda * annualized_turnover
    turnover_penalty_lambda: float = 0.0

    # ML / position scalar controls
    ml_enabled: bool = True
    ml_modular_path_enabled: bool = False
    ml_position_scalar_method: str = "hmm_prob"
    ml_hmm_scalar_floor: float = 0.30
    ml_hmm_scalar_ceiling: float = 1.50
    ml_probability_threshold: float = 0.0
    ml_rf_alpha_weight: float = 1.0
    ml_rf_regime_weight: float = 0.5
    ml_rf_bias: float = 0.0
    ml_position_scalar: float = 1.0
    ml_position_scalar_auto: bool = True
    ml_position_scalar_target_vol: float = 0.10
    ml_position_scalar_max: float = 50.0

    # Template-like ML overlay (matches Template/TEMA-TEMPLATE(NEW_).py semantics)
    # NOTE: This overlay is applied to portfolio return streams, not to per-asset weights.
    ml_template_overlay_enabled: bool = False
    hmm_n_states: int = 2
    hmm_n_iter: int = 30
    hmm_var_floor: float = 1e-8
    hmm_trans_sticky: float = 0.92
    rf_n_estimators: int = 400
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 20
    rf_random_state: int = 42
    ml_prob_threshold: float = 0.55
    ml_auto_threshold: bool = True
    ml_target_exposure: float = 0.10
    # Computed-mode ML overlay tuning (kept disabled for precomputed parity mode)
    ml_computed_overlay_tuning_enabled: bool = True
    ml_computed_overlay_tuning_validation_ratio: float = 0.25
    ml_computed_overlay_tuning_min_rows: int = 80
    ml_computed_overlay_tuning_folds: int = 3
    ml_computed_overlay_tuning_overfit_penalty: float = 0.5
    ml_computed_overlay_grid_rf_n_estimators: tuple[int, ...] = (300, 400)
    ml_computed_overlay_grid_rf_max_depth: tuple[int, ...] = (4, 6)
    ml_computed_overlay_grid_rf_min_samples_leaf: tuple[int, ...] = (20, 40)
    ml_computed_overlay_grid_target_exposure: tuple[float, ...] = (0.10, 0.20, 0.40)
    ml_computed_overlay_grid_hmm_n_states: tuple[int, ...] = (2, 3)

    # Phase1 meta overlay (ML_META): scale ML series by a learned exposure time-series
    ml_meta_overlay_enabled: bool = False
    ml_meta_lags: int = 5
    ml_meta_roll: int = 5
    ml_meta_k_min: float = 0.5
    ml_meta_k_max: float = 8.0
    ml_meta_k_steps: int = 32
    ml_meta_floors: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 0.9)
    ml_meta_min_vol_ratio: float = 0.5
    ml_meta_min_mean_abs_exposure: float = 0.7
    ml_meta_min_turnover_per_year: float = 5.0
    ml_meta_target_mean_abs_exposure: float = 0.9
    ml_meta_target_turnover_per_year: float = 20.0
    # Computed-only ML_META selection shaping (used when template precomputed artifacts are disabled)
    ml_meta_computed_allow_zero_floor: bool = True
    ml_meta_computed_target_mean_abs_exposure: float = 0.10
    ml_meta_computed_target_turnover_per_year: float = 2.0
    ml_meta_computed_exposure_tolerance: float = 0.10
    ml_meta_computed_turnover_tolerance: float = 2.0
    ml_meta_computed_validation_ratio: float = 0.25
    ml_meta_computed_validation_min_rows: int = 80
    ml_meta_computed_validation_folds: int = 3
    ml_meta_computed_overfit_penalty: float = 0.5
    # Optional parity-calibration mode:
    # allow loading benchmark ML_META CSVs even when template precomputed
    # artifacts are disabled (computed pipeline remains default otherwise).
    ml_meta_comparator_use_benchmark_csv: bool = False
    # Honest guardrail: fail-fast when benchmark/comparator CSV data is used
    # in generated run outputs (for example ML_META benchmark CSV injection).
    strict_independent_mode: bool = False

    # Vol-target scaling controls
    vol_target_enabled: bool = True
    vol_target_annual: float = 0.10
    vol_target_max_leverage: float = 12.0
    vol_target_min_leverage: float = 0.25
    vol_target_reference: str = "bl"
    vol_target_apply_to_ml: bool = False

    # Drawdown guard overlay (exposure reduction during sustained drawdowns)
    dd_guard_enabled: bool = False
    dd_guard_max_drawdown: float = 0.10
    dd_guard_floor: float = 0.25
    dd_guard_allow_full_derisk: bool = True
    dd_guard_recovery_halflife: int = 20

    # Confluence Leverage Engine (CLE) controls
    cle_enabled: bool = False
    cle_use_external_proxies: bool = False
    cle_mode: str = "confluence_blend"
    cle_mapping_mode: str = "linear"
    cle_mapping_min_multiplier: float = 0.5
    cle_mapping_max_multiplier: float = 1.5
    cle_mapping_step_thresholds: tuple[float, ...] = (0.30, 0.70)
    cle_mapping_step_multipliers: tuple[float, ...] = (0.50, 1.00, 1.50)
    cle_mapping_kelly_gamma: float = 2.0
    cle_gate_event_blackout_cap: float = 0.5
    cle_gate_liquidity_spread_z_threshold: float = 2.0
    cle_gate_liquidity_depth_threshold: float = 0.10
    cle_gate_liquidity_reduction_factor: float = 0.25
    cle_gate_correlation_alert_cap: float = 1.0
    cle_base_leverage: float = 1.0
    cle_confluence_intercept: float = 0.0
    cle_confluence_temperature: float = 1.0
    cle_confluence_winsor_lower_quantile: float = 0.05
    cle_confluence_winsor_upper_quantile: float = 0.95
    cle_force_event_blackout: bool = False
    cle_event_blackout_dd_threshold: float = -1.0
    cle_spread_z_from_regime_scale: float = 1.0
    cle_spread_z_override: Optional[float] = None
    cle_depth_percentile_override: Optional[float] = None
    cle_force_correlation_alert: bool = False
    cle_correlation_alert_threshold: float = 1.01
    cle_leverage_floor: float = 0.0
    cle_leverage_cap: float = 12.0
    cle_policy_seed: int = 42
    cle_online_calibration_enabled: bool = False
    cle_online_calibration_window: int = 5
    cle_online_calibration_learning_rate: float = 0.10
    cle_online_calibration_l2: float = 1e-4

    # Dynamic ensemble controls (Phase 1)
    ensemble_enabled: bool = False
    ensemble_lookback: int = 20
    ensemble_ridge_shrink: float = 0.15
    ensemble_min_weight: float = 0.05
    ensemble_max_weight: float = 0.90
    ensemble_regime_sensitivity: float = 0.40
    online_learning_enabled: bool = False
    online_learning_learning_rate: float = 0.10
    online_learning_l2: float = 1e-4
    online_learning_seed: int = 42

    # Stress-testing controls (Phase 5)
    stress_enabled: bool = False
    stress_seed: int = 42
    stress_n_paths: int = 200
    stress_horizon: int = 20

    # Costs
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    # Execution cost model settings (neutral defaults preserve legacy behaviour)
    cost_model: str = "simple"
    spread_bps: float = 0.0
    impact_coeff: float = 0.0
    borrow_bps: float = 0.0

    # Generic
    freq: str = "D"
    risk_free_rate: float = 0.0

    # Monitoring / run ledger
    monitoring_ledger_enabled: bool = False
    monitoring_ledger_path: Optional[str] = None

    # When True, and template_default_universe is enabled, the backtest will
    # use a constant final_weights schedule instead of daily signal-derived
    # reweight blending. This reduces execution-path mismatch vs Template.
    backtest_static_weights_in_template: bool = False

    def total_cost_rate(self) -> float:
        return float(self.fee_rate + self.slippage_rate)
