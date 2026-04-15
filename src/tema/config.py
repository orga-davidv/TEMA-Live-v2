from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestConfig:
    # Turnover / rebalance controls (Phase 2b)
    rebalance_min_threshold: float = 0.001
    cost_aware_rebalance: bool = False
    cost_aware_rebalance_multiplier: float = 1.0
    cost_aware_alpha_lookback: int = 20

    # Penalty applied during selection/optimization: Sharpe - lambda * annualized_turnover
    turnover_penalty_lambda: float = 0.0

    # ML / position scalar controls
    ml_enabled: bool = True
    ml_position_scalar_method: str = "hmm_prob"
    ml_hmm_scalar_floor: float = 0.30
    ml_hmm_scalar_ceiling: float = 1.50
    ml_position_scalar: float = 1.0
    ml_position_scalar_auto: bool = True
    ml_position_scalar_target_vol: float = 0.10
    ml_position_scalar_max: float = 50.0

    # Vol-target scaling controls
    vol_target_enabled: bool = True
    vol_target_annual: float = 0.10
    vol_target_max_leverage: float = 12.0
    vol_target_min_leverage: float = 0.25
    vol_target_reference: str = "bl"
    vol_target_apply_to_ml: bool = False

    # Costs
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005

    # Generic
    freq: str = "D"
    
    def total_cost_rate(self) -> float:
        return float(self.fee_rate + self.slippage_rate)
