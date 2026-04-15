from typing import Sequence, List
from .config import BacktestConfig
from .turnover import apply_rebalance_gating


class Runner:
    """Minimal runner exposing modular entrypoints for Wave 1 features.

    The Runner is intentionally small: it demonstrates the path that calls
    cost-aware gating and exposes config-driven ML/vol flags. This is not a
    full backtest runner but a reachable integration surface for tests and
    incremental adoption.
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def decide_portfolio_weights(self, current: Sequence[float], candidate: Sequence[float], expected_alphas: Sequence[float]) -> List[float]:
        """Apply gating (turnover/cost-aware) and return final weights."""
        # If cost-aware gating disabled, apply_rebalance_gating will still allow based on min threshold
        final = apply_rebalance_gating(current, candidate, expected_alphas, self.cfg)
        return final

    def ml_and_vol_flags(self) -> dict:
        return {
            "ml_enabled": self.cfg.ml_enabled,
            "ml_scalar_auto": self.cfg.ml_position_scalar_auto,
            "vol_target_enabled": self.cfg.vol_target_enabled,
            "vol_target_apply_to_ml": self.cfg.vol_target_apply_to_ml,
        }
