from typing import Any, Sequence, List, Tuple
from .config import BacktestConfig


def _periods_per_year(freq: str) -> float:
    f = (freq or "D").upper()
    if f.startswith("D"):
        return 252.0
    if f.startswith("W"):
        return 52.0
    if f.startswith("M"):
        return 12.0
    if f.startswith("H"):
        return 252.0 * 24.0
    return 252.0


def evaluate_rebalance_gate(current_w: float, candidate_w: float, expected_alpha: float, cfg: BacktestConfig) -> dict[str, Any]:
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
        return {
            "allow": True,
            "reason": "no_config",
            "delta_weight": float(abs(candidate_w - current_w)),
            "expected_alpha": float(expected_alpha),
            "annualized_turnover": None,
            "annualized_costs": None,
            "cost_threshold": None,
        }

    delta = abs(candidate_w - current_w)
    out = {
        "allow": False,
        "reason": "unknown",
        "delta_weight": float(delta),
        "expected_alpha": float(expected_alpha),
        "annualized_turnover": None,
        "annualized_costs": None,
        "cost_threshold": None,
    }
    if delta < cfg.rebalance_min_threshold:
        out["reason"] = "below_min_threshold"
        return out

    if not cfg.cost_aware_rebalance:
        out["allow"] = True
        out["reason"] = "cost_aware_disabled"
        return out

    periods = _periods_per_year(cfg.freq)
    annualized_turnover = delta * periods
    cost_rate = cfg.total_cost_rate()
    annual_costs = annualized_turnover * cost_rate

    # compare expected alpha against annual_costs scaled by multiplier
    threshold = annual_costs * cfg.cost_aware_rebalance_multiplier
    out["annualized_turnover"] = float(annualized_turnover)
    out["annualized_costs"] = float(annual_costs)
    out["cost_threshold"] = float(threshold)
    out["allow"] = bool(expected_alpha > threshold)
    out["reason"] = "allowed" if out["allow"] else "cost_gate_blocked"
    return out


def should_rebalance(current_w: float, candidate_w: float, expected_alpha: float, cfg: BacktestConfig) -> bool:
    return bool(evaluate_rebalance_gate(current_w, candidate_w, expected_alpha, cfg).get("allow", False))


def apply_rebalance_gating(current_weights: Sequence[float], candidate_weights: Sequence[float], expected_alphas: Sequence[float], cfg: BacktestConfig) -> List[float]:
    """
    Return a new set of weights where candidate_weights that fail gating are replaced by current_weights.
    Vectorized across assets; lengths must match.
    """
    out, _ = apply_rebalance_gating_with_diagnostics(
        current_weights=current_weights,
        candidate_weights=candidate_weights,
        expected_alphas=expected_alphas,
        cfg=cfg,
    )
    return out


def apply_rebalance_gating_with_diagnostics(
    current_weights: Sequence[float],
    candidate_weights: Sequence[float],
    expected_alphas: Sequence[float],
    cfg: BacktestConfig,
) -> Tuple[List[float], dict[str, Any]]:
    """Vectorized gating with per-asset diagnostics for parity/OOS analysis."""
    if len(current_weights) != len(candidate_weights) or len(current_weights) != len(expected_alphas):
        raise ValueError("Input sequences must have the same length")

    out = []
    per_asset: list[dict[str, Any]] = []
    threshold_blocks = 0
    cost_blocks = 0
    for cw, nw, ea in zip(current_weights, candidate_weights, expected_alphas):
        decision = evaluate_rebalance_gate(cw, nw, ea, cfg)
        per_asset.append(decision)
        if decision["allow"]:
            out.append(nw)
        else:
            out.append(cw)
            if decision["reason"] == "below_min_threshold":
                threshold_blocks += 1
            elif decision["reason"] == "cost_gate_blocked":
                cost_blocks += 1
    diagnostics = {
        "assets": int(len(out)),
        "allowed_count": int(len(out) - threshold_blocks - cost_blocks),
        "threshold_block_count": int(threshold_blocks),
        "cost_block_count": int(cost_blocks),
        "per_asset": per_asset,
    }
    return out, diagnostics
