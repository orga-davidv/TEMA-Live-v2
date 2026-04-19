# tema package init
from .config import BacktestConfig
from .turnover import should_rebalance, apply_rebalance_gating
from .runner import Runner
from .optimization import run_bayesian_optimization, run_and_write_optimization, compute_objective
from .ensemble import DynamicEnsembleConfig, compute_dynamic_ensemble_weights, combine_stream_signals
from .interactions import InteractionScore, compute_pairwise_interaction_scores, select_top_k_interactions, generate_feature_crosses
from .online_learning import OnlineLearningConfig, OnlineLogisticLearner
from .stress import compute_scenario_metrics, evaluate_stress_scenarios, historical_shock_scenarios, sample_scenario_paths
from .ml import score_regime_probabilities, score_rf_probabilities, threshold_probabilities, compute_position_scalars
from .portfolio import allocate_portfolio_weights, PortfolioAllocationResult, hrp_allocation_hook, nco_allocation_hook
from .backtest import run_return_equity_simulation, compute_backtest_metrics, build_weight_schedule_from_signals, BacktestResult
from .external import (
    ExternalProxyLoadResult,
    load_proxy_from_csv,
    load_proxy_from_stub,
    load_proxy_adapter,
    load_macro_proxy_adapter,
    load_calendar_proxy_adapter,
    load_liquidity_proxy_adapter,
    load_cle_external_proxies,
)

__all__ = [
    "BacktestConfig",
    "should_rebalance",
    "apply_rebalance_gating",
    "Runner",
    "run_bayesian_optimization",
    "run_and_write_optimization",
    "compute_objective",
    "DynamicEnsembleConfig",
    "compute_dynamic_ensemble_weights",
    "combine_stream_signals",
    "InteractionScore",
    "compute_pairwise_interaction_scores",
    "select_top_k_interactions",
    "generate_feature_crosses",
    "OnlineLearningConfig",
    "OnlineLogisticLearner",
    "compute_scenario_metrics",
    "historical_shock_scenarios",
    "sample_scenario_paths",
    "evaluate_stress_scenarios",
    "score_regime_probabilities",
    "score_rf_probabilities",
    "threshold_probabilities",
    "compute_position_scalars",
    "allocate_portfolio_weights",
    "PortfolioAllocationResult",
    "hrp_allocation_hook",
    "nco_allocation_hook",
    "run_return_equity_simulation",
    "compute_backtest_metrics",
    "build_weight_schedule_from_signals",
    "BacktestResult",
    "ExternalProxyLoadResult",
    "load_proxy_from_csv",
    "load_proxy_from_stub",
    "load_proxy_adapter",
    "load_macro_proxy_adapter",
    "load_calendar_proxy_adapter",
    "load_liquidity_proxy_adapter",
    "load_cle_external_proxies",
]
