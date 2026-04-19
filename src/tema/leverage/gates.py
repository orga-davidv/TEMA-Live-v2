from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardGateConfig:
    event_blackout_cap: float = 0.5
    liquidity_spread_z_threshold: float = 2.0
    liquidity_depth_threshold: float = 0.10
    liquidity_reduction_factor: float = 0.25
    correlation_alert_cap: float = 1.0

    def __post_init__(self) -> None:
        if float(self.event_blackout_cap) < 0.0:
            raise ValueError("event_blackout_cap must be >= 0")
        if float(self.liquidity_reduction_factor) < 0.0:
            raise ValueError("liquidity_reduction_factor must be >= 0")
        if float(self.correlation_alert_cap) < 0.0:
            raise ValueError("correlation_alert_cap must be >= 0")
        if not 0.0 <= float(self.liquidity_depth_threshold) <= 1.0:
            raise ValueError("liquidity_depth_threshold must be in [0, 1]")


def apply_event_blackout_cap(leverage: float, event_blackout: bool, cap: float = 0.5) -> float:
    """Cap leverage during event blackout windows."""
    lev = max(0.0, float(leverage))
    if not bool(event_blackout):
        return lev
    return float(min(lev, max(0.0, float(cap))))


def apply_liquidity_gate(
    leverage: float,
    *,
    spread_z: float | None,
    depth_percentile: float | None,
    spread_z_threshold: float = 2.0,
    depth_threshold: float = 0.10,
    reduction_factor: float = 0.25,
) -> float:
    """Reduce leverage when spread or depth indicates poor liquidity."""
    lev = max(0.0, float(leverage))
    spread_trigger = spread_z is not None and float(spread_z) > float(spread_z_threshold)
    depth_trigger = depth_percentile is not None and float(depth_percentile) < float(depth_threshold)
    if not (spread_trigger or depth_trigger):
        return lev
    factor = max(0.0, float(reduction_factor))
    return float(lev * factor)


def apply_correlation_alert_cap(leverage: float, correlation_alert: bool, cap: float = 1.0) -> float:
    """Cap leverage when correlation-breakdown alert is active."""
    lev = max(0.0, float(leverage))
    if not bool(correlation_alert):
        return lev
    return float(min(lev, max(0.0, float(cap))))


def apply_hard_gates(
    leverage: float,
    *,
    event_blackout: bool = False,
    spread_z: float | None = None,
    depth_percentile: float | None = None,
    correlation_alert: bool = False,
    cfg: HardGateConfig | None = None,
) -> tuple[float, dict[str, bool]]:
    """Apply all hard gates in fixed order and return flags for diagnostics."""
    eff = cfg if cfg is not None else HardGateConfig()

    lev = max(0.0, float(leverage))
    event_triggered = bool(event_blackout)
    lev = apply_event_blackout_cap(lev, event_triggered, cap=eff.event_blackout_cap)

    spread_trigger = spread_z is not None and float(spread_z) > float(eff.liquidity_spread_z_threshold)
    depth_trigger = depth_percentile is not None and float(depth_percentile) < float(eff.liquidity_depth_threshold)
    liquidity_triggered = bool(spread_trigger or depth_trigger)
    lev = apply_liquidity_gate(
        lev,
        spread_z=spread_z,
        depth_percentile=depth_percentile,
        spread_z_threshold=eff.liquidity_spread_z_threshold,
        depth_threshold=eff.liquidity_depth_threshold,
        reduction_factor=eff.liquidity_reduction_factor,
    )

    corr_triggered = bool(correlation_alert)
    lev = apply_correlation_alert_cap(lev, corr_triggered, cap=eff.correlation_alert_cap)

    return float(np.clip(lev, 0.0, np.inf)), {
        "event_blackout": event_triggered,
        "liquidity": liquidity_triggered,
        "correlation_alert": corr_triggered,
    }
