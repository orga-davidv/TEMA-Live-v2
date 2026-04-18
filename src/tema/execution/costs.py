from typing import Optional, Sequence
import numpy as np


# Simple, small CostModel interface: a compute function
# compute_cost(turnover, prev_weights, cur_weights, exposure, fee_rate, slippage_rate, spread_bps, impact_coeff, borrow_bps)

def compute_transaction_cost(
    turnover: float,
    prev_weights: Optional[Sequence[float]],
    cur_weights: Sequence[float],
    exposure: Optional[Sequence[float]] = None,
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    cost_model: str = "simple",
    spread_bps: float = 0.0,
    impact_coeff: float = 0.0,
    borrow_bps: float = 0.0,
) -> float:
    """
    Compute transaction cost for a single period.

    - turnover: sum absolute weight changes (0..2 for full flip)
    - prev_weights, cur_weights: previous and current executed weights (may be None for first period)
    - exposure: optional per-asset exposures (weights) to compute borrow fees
    - fee_rate, slippage_rate: legacy per-unit-turnover rates (decimal)
    - cost_model: currently supports 'simple' and 'extended'
    - spread_bps: spread in basis points (bps). Interpreted as round-trip bps and converted to decimal.
    - impact_coeff: simple linear impact coefficient; applied as impact_coeff * turnover^2
    - borrow_bps: bps charged on net short exposure per period (decimal: bps -> /10000)

    Returns total cost (in same units as PnL, i.e., fraction of portfolio value).
    """
    # Legacy simple model: cost = turnover * (fee + slippage)
    total_cost = 0.0
    legacy_rate = float(fee_rate + slippage_rate)
    if cost_model in (None, "simple"):
        total_cost = turnover * legacy_rate
        return float(total_cost)

    # Extended model: start with legacy component
    total_cost = turnover * legacy_rate

    # Spread: treat spread_bps as round-trip in basis points -> convert to decimal
    try:
        spread_dec = float(spread_bps) / 10000.0
    except Exception:
        spread_dec = 0.0
    if spread_dec and turnover > 0.0:
        # Apply spread as proportional to turnover (round-trip cost)
        total_cost += turnover * spread_dec

    # Market impact: simple quadratic-ish model: impact_coeff * turnover^2
    try:
        ic = float(impact_coeff)
    except Exception:
        ic = 0.0
    if ic and turnover > 0.0:
        total_cost += float(ic) * (float(turnover) ** 2)

    # Borrow fees: apply to negative exposure (shorts) as borrow_bps per period
    try:
        bb = float(borrow_bps) / 10000.0
    except Exception:
        bb = 0.0
    if bb and exposure is not None:
        exp = np.asarray(exposure, dtype=float)
        short_exposure = float(np.sum(np.clip(-exp, 0.0, None)))
        if short_exposure > 0.0:
            total_cost += bb * short_exposure

    return float(total_cost)
