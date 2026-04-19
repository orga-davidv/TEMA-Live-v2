from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_ALLOWED_MAPPING_MODES = {"linear", "stepwise", "kelly_shrink"}


@dataclass(frozen=True)
class ConfluenceMappingConfig:
    mode: str = "linear"
    min_multiplier: float = 0.5
    max_multiplier: float = 1.5
    step_thresholds: tuple[float, ...] = (0.30, 0.70)
    step_multipliers: tuple[float, ...] = (0.50, 1.00, 1.50)
    kelly_gamma: float = 2.0

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in _ALLOWED_MAPPING_MODES:
            raise ValueError(f"mode must be one of {_ALLOWED_MAPPING_MODES}")
        if float(self.min_multiplier) < 0.0 or float(self.max_multiplier) < 0.0:
            raise ValueError("multipliers must be non-negative")
        if float(self.min_multiplier) > float(self.max_multiplier):
            raise ValueError("min_multiplier must be <= max_multiplier")
        if float(self.kelly_gamma) <= 0.0:
            raise ValueError("kelly_gamma must be > 0")

        thresholds = tuple(float(x) for x in self.step_thresholds)
        if any(x < 0.0 or x > 1.0 for x in thresholds):
            raise ValueError("step_thresholds values must be in [0, 1]")
        if any(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError("step_thresholds must be strictly increasing")

        if len(self.step_multipliers) != len(self.step_thresholds) + 1:
            raise ValueError("step_multipliers length must be len(step_thresholds) + 1")
        if any(float(x) < 0.0 for x in self.step_multipliers):
            raise ValueError("step_multipliers must be non-negative")


def _clip_score(score: float) -> float:
    return float(np.clip(float(score), 0.0, 1.0))


def _clip_multiplier(value: float, cfg: ConfluenceMappingConfig) -> float:
    return float(np.clip(float(value), float(cfg.min_multiplier), float(cfg.max_multiplier)))


def _linear_multiplier(score: float, cfg: ConfluenceMappingConfig) -> float:
    s = _clip_score(score)
    return float(cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * s)


def _stepwise_multiplier(score: float, thresholds: Sequence[float], levels: Sequence[float]) -> float:
    s = _clip_score(score)
    idx = 0
    for threshold in thresholds:
        if s < float(threshold):
            break
        idx += 1
    return float(levels[idx])


def _kelly_shrink_multiplier(score: float, cfg: ConfluenceMappingConfig) -> float:
    s = _clip_score(score)
    shrink = s ** float(cfg.kelly_gamma)
    return float(cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * shrink)


def map_confluence_to_multiplier(score: float, cfg: ConfluenceMappingConfig | None = None) -> float:
    """Map confluence score in [0,1] to a leverage multiplier."""
    eff = cfg if cfg is not None else ConfluenceMappingConfig()
    mode = str(eff.mode).lower()

    if mode == "linear":
        raw = _linear_multiplier(score, eff)
    elif mode == "stepwise":
        raw = _stepwise_multiplier(score, eff.step_thresholds, eff.step_multipliers)
    elif mode == "kelly_shrink":
        raw = _kelly_shrink_multiplier(score, eff)
    else:
        raise ValueError(f"Unsupported mode: {eff.mode}")

    return _clip_multiplier(raw, eff)


def compute_confluence_multiplier(score: float, cfg: ConfluenceMappingConfig | None = None) -> float:
    """Alias for map_confluence_to_multiplier."""
    return map_confluence_to_multiplier(score=score, cfg=cfg)
