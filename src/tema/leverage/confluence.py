from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConfluenceConfig:
    winsor_lower_quantile: float = 0.05
    winsor_upper_quantile: float = 0.95
    zscore_epsilon: float = 1e-12
    intercept: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.winsor_lower_quantile) <= 1.0:
            raise ValueError("winsor_lower_quantile must be in [0, 1]")
        if not 0.0 <= float(self.winsor_upper_quantile) <= 1.0:
            raise ValueError("winsor_upper_quantile must be in [0, 1]")
        if float(self.winsor_lower_quantile) > float(self.winsor_upper_quantile):
            raise ValueError("winsor_lower_quantile must be <= winsor_upper_quantile")
        if float(self.zscore_epsilon) <= 0.0:
            raise ValueError("zscore_epsilon must be > 0")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be > 0")


def _as_1d_float_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def winsorize(values: Sequence[float], lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> np.ndarray:
    """Quantile-clipped vector with deterministic finite output."""
    if not 0.0 <= float(lower_quantile) <= 1.0:
        raise ValueError("lower_quantile must be in [0, 1]")
    if not 0.0 <= float(upper_quantile) <= 1.0:
        raise ValueError("upper_quantile must be in [0, 1]")
    if float(lower_quantile) > float(upper_quantile):
        raise ValueError("lower_quantile must be <= upper_quantile")

    arr = _as_1d_float_array(values)
    if arr.size == 0:
        return arr

    lo = float(np.quantile(arr, float(lower_quantile)))
    hi = float(np.quantile(arr, float(upper_quantile)))
    return np.clip(arr, lo, hi)


def zscore(values: Sequence[float], epsilon: float = 1e-12) -> np.ndarray:
    """Standard-score normalization with zero-vector fallback."""
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be > 0")
    arr = _as_1d_float_array(values)
    if arr.size == 0:
        return arr

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= float(epsilon):
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def align_sign(values: Sequence[float], sign: float) -> np.ndarray:
    """Align signal direction: positive sign keeps values, negative sign flips."""
    if float(sign) == 0.0:
        raise ValueError("sign must be non-zero")
    direction = 1.0 if float(sign) > 0.0 else -1.0
    arr = _as_1d_float_array(values)
    return arr * direction


def _sigmoid(value: float) -> float:
    bounded = float(np.clip(float(value), -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-bounded))


def _build_confluence_diagnostics(
    *,
    keys: Sequence[str],
    raw_values: np.ndarray,
    aligned_values: np.ndarray,
    normalized_values: np.ndarray,
    raw_weight_values: np.ndarray,
    normalized_weight_values: np.ndarray,
    contributions: np.ndarray,
    intercept: float,
    temperature: float,
    weighted_sum: float,
    logit: float,
    score: float,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for idx, key in enumerate(keys):
        components.append(
            {
                "name": str(key),
                "raw_signal": float(raw_values[idx]),
                "aligned_signal": float(aligned_values[idx]),
                "normalized_signal": float(normalized_values[idx]),
                "weight_raw": float(raw_weight_values[idx]),
                "weight": float(normalized_weight_values[idx]),
                "contribution": float(contributions[idx]),
            }
        )
    return {
        "components": components,
        "weighted_sum": float(weighted_sum),
        "intercept": float(intercept),
        "temperature": float(temperature),
        "logit": float(logit),
        "score": float(score),
    }


def compute_confluence_score(
    signals: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    sign_map: Mapping[str, float] | None = None,
    cfg: ConfluenceConfig | None = None,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Aggregate aligned/normalized signals into a deterministic score in [0, 1]."""
    eff = cfg if cfg is not None else ConfluenceConfig()
    if signals is None or len(signals) == 0:
        score = 0.5
        if not return_diagnostics:
            return score
        diagnostics = _build_confluence_diagnostics(
            keys=[],
            raw_values=np.empty((0,), dtype=float),
            aligned_values=np.empty((0,), dtype=float),
            normalized_values=np.empty((0,), dtype=float),
            raw_weight_values=np.empty((0,), dtype=float),
            normalized_weight_values=np.empty((0,), dtype=float),
            contributions=np.empty((0,), dtype=float),
            intercept=float(eff.intercept),
            temperature=float(eff.temperature),
            weighted_sum=0.0,
            logit=float(eff.intercept),
            score=float(score),
        )
        diagnostics["reason"] = "empty_signals"
        return float(score), diagnostics

    keys = sorted(str(k) for k in signals.keys())

    raw_values = []
    aligned_values = []
    weight_values = []
    for key in keys:
        raw_value = float(signals[key])
        if not np.isfinite(raw_value):
            raw_value = 0.0
        raw_values.append(raw_value)

        direction = 1.0
        if sign_map is not None and key in sign_map:
            direction = float(sign_map[key])
        aligned_values.append(float(align_sign([raw_value], direction)[0]))

        if weights is None:
            weight_values.append(1.0)
        else:
            w = float(weights.get(key, 1.0))
            weight_values.append(w if np.isfinite(w) else 0.0)

    raw_arr = np.asarray(raw_values, dtype=float)
    aligned = np.asarray(aligned_values, dtype=float)
    normalized = zscore(
        winsorize(
            aligned,
            lower_quantile=float(eff.winsor_lower_quantile),
            upper_quantile=float(eff.winsor_upper_quantile),
        ),
        epsilon=float(eff.zscore_epsilon),
    )

    weight_vec = np.asarray(weight_values, dtype=float)
    denom = float(np.sum(np.abs(weight_vec)))
    if denom <= float(eff.zscore_epsilon):
        weight_vec = np.ones_like(weight_vec, dtype=float) / float(weight_vec.size)
    else:
        weight_vec = weight_vec / denom

    weighted_sum = float(np.dot(weight_vec, normalized))
    contributions = weight_vec * normalized
    logit = float(eff.intercept + weighted_sum)
    score = _sigmoid(logit / float(eff.temperature))
    clipped_score = float(np.clip(score, 0.0, 1.0))
    if not return_diagnostics:
        return clipped_score
    diagnostics = _build_confluence_diagnostics(
        keys=keys,
        raw_values=raw_arr,
        aligned_values=aligned,
        normalized_values=normalized,
        raw_weight_values=np.asarray(weight_values, dtype=float),
        normalized_weight_values=weight_vec,
        contributions=contributions,
        intercept=float(eff.intercept),
        temperature=float(eff.temperature),
        weighted_sum=float(weighted_sum),
        logit=float(logit),
        score=float(clipped_score),
    )
    return clipped_score, diagnostics
