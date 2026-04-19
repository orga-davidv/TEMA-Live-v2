from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gates import HardGateConfig, apply_hard_gates
from .mapping import ConfluenceMappingConfig, map_confluence_to_multiplier


@dataclass(frozen=True)
class LeverageEngineConfig:
    mapping: ConfluenceMappingConfig = field(default_factory=ConfluenceMappingConfig)
    gates: HardGateConfig = field(default_factory=HardGateConfig)
    leverage_floor: float = 0.0
    leverage_cap: float = 12.0

    def __post_init__(self) -> None:
        if float(self.leverage_floor) < 0.0:
            raise ValueError("leverage_floor must be >= 0")
        if float(self.leverage_cap) < 0.0:
            raise ValueError("leverage_cap must be >= 0")
        if float(self.leverage_floor) > float(self.leverage_cap):
            raise ValueError("leverage_floor must be <= leverage_cap")


def compute_leverage(
    base_leverage: float,
    confluence_score: float,
    *,
    cfg: LeverageEngineConfig | None = None,
    event_blackout: bool = False,
    spread_z: float | None = None,
    depth_percentile: float | None = None,
    correlation_alert: bool = False,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Compute final leverage from base leverage and confluence controls."""
    eff = cfg if cfg is not None else LeverageEngineConfig()

    base = float(base_leverage)
    if not np.isfinite(base):
        raise ValueError("base_leverage must be finite")
    base = max(0.0, base)

    multiplier = map_confluence_to_multiplier(score=float(confluence_score), cfg=eff.mapping)
    pre_gate = float(base * multiplier)

    gated, gate_flags = apply_hard_gates(
        pre_gate,
        event_blackout=event_blackout,
        spread_z=spread_z,
        depth_percentile=depth_percentile,
        correlation_alert=correlation_alert,
        cfg=eff.gates,
    )
    final = float(np.clip(gated, float(eff.leverage_floor), float(eff.leverage_cap)))

    if not return_diagnostics:
        return final

    diagnostics = {
        "base_leverage": float(base),
        "confluence_score": float(np.clip(float(confluence_score), 0.0, 1.0)),
        "multiplier": float(multiplier),
        "pre_gate_leverage": float(pre_gate),
        "post_gate_leverage": float(gated),
        "final_leverage": float(final),
        "gate_flags": gate_flags,
        "leverage_floor": float(eff.leverage_floor),
        "leverage_cap": float(eff.leverage_cap),
        "mapping_mode": str(eff.mapping.mode),
        "mapping_parameters": {
            "min_multiplier": float(eff.mapping.min_multiplier),
            "max_multiplier": float(eff.mapping.max_multiplier),
            "step_thresholds": [float(x) for x in eff.mapping.step_thresholds],
            "step_multipliers": [float(x) for x in eff.mapping.step_multipliers],
            "kelly_gamma": float(eff.mapping.kelly_gamma),
        },
        "gate_parameters": {
            "event_blackout_cap": float(eff.gates.event_blackout_cap),
            "liquidity_spread_z_threshold": float(eff.gates.liquidity_spread_z_threshold),
            "liquidity_depth_threshold": float(eff.gates.liquidity_depth_threshold),
            "liquidity_reduction_factor": float(eff.gates.liquidity_reduction_factor),
            "correlation_alert_cap": float(eff.gates.correlation_alert_cap),
        },
    }
    return final, diagnostics
