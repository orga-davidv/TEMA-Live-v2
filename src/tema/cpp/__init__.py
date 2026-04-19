"""
C++ bridge shim for TEMA
- Detects available compiled C++ wrappers
- Exposes get_signal_engine() returning an object compatible with SignalEngine
- Deterministic Python fallback when C++ unavailable

Design:
- Look for commonly named shared libraries in Template/ or build/ directories
- If found, attempt to load with ctypes and expose simple functions
- API: has_cpp() -> bool, get_signal_engine() -> SignalEngine-like object
"""
from __future__ import annotations

import os
import ctypes
from typing import Any
import logging

import pandas as pd

# Keep import light to avoid heavy deps or circular imports
try:
    from tema.signals.engine import SignalEngine, PythonSignalEngine
    from tema.signals.tema import generate_crossover_signal_matrix
except Exception:  # pragma: no cover - defensive
    SignalEngine = Any
    PythonSignalEngine = None

logger = logging.getLogger(__name__)

# Candidate shared lib names and locations (non-exhaustive)
_CANDIDATE_LIB_PATHS = [
    os.path.join("Template", "libgrid_signals.so"),
    os.path.join("Template", "grid_signals.so"),
    os.path.join("Template", "grid_signals.dll"),
    os.path.join("Template", "build", "grid_signals.so"),
    os.path.join("build", "grid_signals.so"),
]

_has_cpp = None
_cpp_lib = None


def _try_load_cpp() -> bool:
    global _has_cpp, _cpp_lib
    if _has_cpp is not None:
        return _has_cpp

    for p in _CANDIDATE_LIB_PATHS:
        if os.path.exists(p):
            try:
                _cpp_lib = ctypes.CDLL(os.path.abspath(p))
                _has_cpp = True
                logger.info("Loaded C++ bridge: %s", p)
                return True
            except Exception:
                logger.exception("Found candidate C++ library but failed to load: %s", p)
    # Attempt to load by module name if installed as python-ext (best-effort)
    try:
        import importlib

        mod = importlib.import_module("grid_signals")
        # if module loaded, we consider C++ available (e.g., pybind11 module)
        _cpp_lib = mod
        _has_cpp = True
        logger.info("Loaded C++ pybind11 module 'grid_signals'")
        return True
    except Exception:
        pass

    _has_cpp = False
    return False


def has_cpp() -> bool:
    """Return True if a C++ bridge implementation was found and loaded."""
    return _try_load_cpp()


class _CppSignalEngine:
    """Signal engine that delegates to a C++ shared library when available.

    The C++ library is expected to provide a C-compatible function with a
    simple interface. Because binary interfaces vary, this class attempts to
    call a few common entry points. If none are found at runtime the class
    will raise a RuntimeError so callers can fall back.
    """

    def __init__(self, lib):
        self.lib = lib

    def generate(self, price_df: pd.DataFrame, fast_period: int, slow_period: int, method: str) -> pd.DataFrame:
        """Generate signals by delegating to C++ where possible.

        Fallback: call the pure-Python implementation to guarantee deterministic behavior.
        """
        # If lib is a python module that exposes generate_signals, try calling it
        try:
            if hasattr(self.lib, "generate_signals"):
                # Expected signature: (ndarray prices, int fast, int slow, str method) -> 2D ndarray
                prices = price_df.values
                out = self.lib.generate_signals(prices, int(fast_period), int(slow_period), method)
                # Convert to DataFrame with same index/columns
                return pd.DataFrame(out, index=price_df.index, columns=price_df.columns)

            # If lib is a ctypes.CDLL, attempt to call a C function named 'generate_signals'
            if hasattr(self.lib, "generate_signals") and isinstance(self.lib, ctypes.CDLL):
                fn = getattr(self.lib, "generate_signals")
                # We can't safely marshal complex types without a stable ABI; skip and raise
                raise RuntimeError("CTypes-based C++ bridge not supported in this shim without ABI spec")

        except Exception:
            logger.exception("C++ bridge call failed; falling back to Python implementation")

        # Deterministic fallback to Python implementation
        return generate_crossover_signal_matrix(
            price_df=price_df,
            fast_period=fast_period,
            slow_period=slow_period,
            method=method,
            shift_by=0,
        )


def get_signal_engine(prefer_cpp: bool = True) -> SignalEngine:
    """Return a SignalEngine. If prefer_cpp True and C++ bridge available, return C++ engine.
    Otherwise return PythonSignalEngine fallback. This call is deterministic and never fails.
    """
    try:
        if prefer_cpp and has_cpp() and _cpp_lib is not None:
            try:
                return _CppSignalEngine(_cpp_lib)
            except Exception:
                logger.exception("Failed to instantiate C++ signal engine; using Python fallback")
        # Fallback path
        if PythonSignalEngine is not None:
            return PythonSignalEngine()
    except Exception:
        logger.exception("Unexpected error creating signal engine; returning a basic Python fallback")
    # As a last resort create a simple lambda-based fallback
    class _BareFallback:
        def generate(self, price_df, fast_period, slow_period, method):
            return generate_crossover_signal_matrix(
                price_df=price_df,
                fast_period=fast_period,
                slow_period=slow_period,
                method=method,
                shift_by=0,
            )

    return _BareFallback()
