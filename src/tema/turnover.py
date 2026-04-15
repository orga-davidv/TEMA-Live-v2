from typing import Sequence, List
import numpy as np
from .config import BacktestConfig


def should_rebalance(current_w: float, candidate_w: float, expected_alpha: float, cfg: BacktestConfig) -> bool:
    """
    Decide whether to perform a rebalance for a single asset weight change.

    Rules implemented (wave 1, conservative):
    - If absolute fractional change < rebalance_min_threshold => do not rebalance
    - If cost-aware gating enabled: require expected_alpha > annualized_costs * multiplier
      where annualized_costs is approximated as cost_rate * abs(delta_weight) * periods_per_year(cfg.freq)

    Note: This is intentionally conservative and local to a single asset; integrating with
    optimization/portfolio-level logic is left for subsequent waves.
    """
    if cfg is None:
        return True

    delta = abs(candidate_w - current_w)
    if delta < cfg.rebalance_min_threshold:
        return False

    if not cfg.cost_aware_rebalance:
        return True

    # approximate annualized turnover by scaling delta by periods per year
    periods = 252.0 if (cfg.freq or "D").upper().startswith("D") else 252.0
    annualized_turnover = delta * periods
    cost_rate = cfg.total_cost_rate()
    annual_costs = annualized_turnover * cost_rate

    # compare expected alpha against annual_costs scaled by multiplier
    threshold = annual_costs * cfg.cost_aware_rebalance_multiplier
    return expected_alpha > threshold


def apply_rebalance_gating(current_weights: Sequence[float], candidate_weights: Sequence[float], expected_alphas: Sequence[float], cfg: BacktestConfig) -> List[float]:
    """
    Return a new set of weights where candidate_weights that fail gating are replaced by current_weights.
    Vectorized across assets; lengths must match.
    """
    if len(current_weights) != len(candidate_weights) or len(current_weights) != len(expected_alphas):
        raise ValueError("Input sequences must have the same length")

    out = []
    for cw, nw, ea in zip(current_weights, candidate_weights, expected_alphas):
        if should_rebalance(cw, nw, ea, cfg):
            out.append(nw)
        else:
            out.append(cw)
    return out
