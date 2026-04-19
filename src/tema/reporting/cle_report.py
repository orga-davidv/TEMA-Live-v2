from __future__ import annotations

from typing import Any, Mapping


CLE_REPORT_SCHEMA_VERSION = "cle_report.v1"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_triggered_gates(
    *,
    gate_flags: Mapping[str, Any],
    gate_context: Mapping[str, Any],
    gate_parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    triggered: list[dict[str, Any]] = []

    if bool(gate_flags.get("event_blackout", False)):
        triggered.append(
            {
                "gate": "event_blackout",
                "reason": "event_blackout flag is active",
                "context": {
                    "event_blackout": bool(gate_context.get("event_blackout", False)),
                    "event_blackout_cap": _as_float(gate_parameters.get("event_blackout_cap"), default=0.0),
                },
            }
        )

    if bool(gate_flags.get("liquidity", False)):
        reasons: list[str] = []
        spread = gate_context.get("spread_z")
        depth = gate_context.get("depth_percentile")
        spread_threshold = gate_parameters.get("liquidity_spread_z_threshold")
        depth_threshold = gate_parameters.get("liquidity_depth_threshold")
        if spread is not None and spread_threshold is not None and _as_float(spread) > _as_float(spread_threshold):
            reasons.append("spread_z above threshold")
        if depth is not None and depth_threshold is not None and _as_float(depth) < _as_float(depth_threshold):
            reasons.append("depth_percentile below threshold")
        if not reasons:
            reasons = ["liquidity gate triggered"]
        triggered.append(
            {
                "gate": "liquidity",
                "reason": ", ".join(reasons),
                "context": {
                    "spread_z": _as_float(spread) if spread is not None else None,
                    "spread_threshold": _as_float(spread_threshold) if spread_threshold is not None else None,
                    "depth_percentile": _as_float(depth) if depth is not None else None,
                    "depth_threshold": _as_float(depth_threshold) if depth_threshold is not None else None,
                    "liquidity_reduction_factor": _as_float(
                        gate_parameters.get("liquidity_reduction_factor"),
                        default=1.0,
                    ),
                },
            }
        )

    if bool(gate_flags.get("correlation_alert", False)):
        triggered.append(
            {
                "gate": "correlation_alert",
                "reason": "correlation_alert flag is active",
                "context": {
                    "correlation_alert": bool(gate_context.get("correlation_alert", False)),
                    "correlation_alert_cap": _as_float(gate_parameters.get("correlation_alert_cap"), default=0.0),
                },
            }
        )
    return triggered


def build_cle_report(leverage_info: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = leverage_info if isinstance(leverage_info, Mapping) else {}
    engine_diag = payload.get("engine_diagnostics") if isinstance(payload.get("engine_diagnostics"), Mapping) else {}
    confluence_diag = payload.get("confluence_diagnostics") if isinstance(payload.get("confluence_diagnostics"), Mapping) else {}
    gate_flags = engine_diag.get("gate_flags") if isinstance(engine_diag.get("gate_flags"), Mapping) else {}
    gate_context = payload.get("gate_context") if isinstance(payload.get("gate_context"), Mapping) else {}
    gate_parameters = engine_diag.get("gate_parameters") if isinstance(engine_diag.get("gate_parameters"), Mapping) else {}

    components = confluence_diag.get("components") if isinstance(confluence_diag.get("components"), list) else []
    component_contributions = [
        {
            "name": str(component.get("name", "")),
            "raw_signal": _as_float(component.get("raw_signal")),
            "aligned_signal": _as_float(component.get("aligned_signal")),
            "normalized_signal": _as_float(component.get("normalized_signal")),
            "weight": _as_float(component.get("weight")),
            "contribution": _as_float(component.get("contribution")),
        }
        for component in components
        if isinstance(component, Mapping)
    ]

    report = {
        "schema_version": CLE_REPORT_SCHEMA_VERSION,
        "enabled": bool(payload.get("enabled", False)),
        "reason": str(payload.get("reason", "")) if payload.get("reason") is not None else "",
        "C_t": _as_float(payload.get("confluence_score"), default=0.5),
        "m_t": _as_float(engine_diag.get("multiplier"), default=1.0),
        "base_leverage": _as_float(payload.get("base_leverage"), default=1.0),
        "pre_gate_leverage": _as_float(engine_diag.get("pre_gate_leverage"), default=1.0),
        "final_leverage": _as_float(payload.get("leverage_scalar"), default=1.0),
        "component_contributions": component_contributions,
        "triggered_gates": _extract_triggered_gates(
            gate_flags=gate_flags,
            gate_context=gate_context,
            gate_parameters=gate_parameters,
        ),
        "gate_flags": {
            "event_blackout": bool(gate_flags.get("event_blackout", False)),
            "liquidity": bool(gate_flags.get("liquidity", False)),
            "correlation_alert": bool(gate_flags.get("correlation_alert", False)),
        },
    }
    return report
