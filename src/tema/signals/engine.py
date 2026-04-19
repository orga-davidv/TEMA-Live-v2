from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .tema import generate_crossover_signal_matrix


class SignalEngine(Protocol):
    def generate(
        self,
        price_df: pd.DataFrame,
        fast_period: int,
        slow_period: int,
        method: str,
    ) -> pd.DataFrame: ...


@dataclass
class PythonSignalEngine:
    # Keep signals unshifted here; execution applies the single walk-forward lag.
    shift_by: int = 0

    def generate(
        self,
        price_df: pd.DataFrame,
        fast_period: int,
        slow_period: int,
        method: str,
    ) -> pd.DataFrame:
        return generate_crossover_signal_matrix(
            price_df=price_df,
            fast_period=fast_period,
            slow_period=slow_period,
            method=method,
            shift_by=self.shift_by,
        )


def resolve_signal_engine(use_cpp: bool = False, cpp_engine: SignalEngine | None = None) -> SignalEngine:
    """Resolve a SignalEngine implementation.

    If use_cpp is True and a cpp_engine is provided, return it. If use_cpp is True
    but cpp_engine is None, attempt to construct a C++ backed engine via the tema.cpp
    bridge. If that fails or use_cpp is False, return the pure Python engine.
    """
    if use_cpp:
        if cpp_engine is not None:
            return cpp_engine
        # Try to construct a C++ engine via the bridge. Fall back deterministically.
        try:
            from tema.cpp import get_signal_engine

            return get_signal_engine(prefer_cpp=True)
        except Exception:
            # If anything goes wrong, we fall back to PythonSignalEngine
            return PythonSignalEngine()
    # Default path: Python implementation
    return PythonSignalEngine()
