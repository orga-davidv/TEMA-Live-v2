from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _default_cpp_source_path() -> Path:
    return Path(__file__).resolve().parent / "cpp" / "hmm_regime.cpp"


def compile_cpp_hmm_library(cpp_path: Path | None = None) -> Path:
    """Compile the C++ HMM helper into a shared library.

    The output is written into a user cache dir to avoid polluting the git working tree.
    """
    cpp_path = Path(cpp_path) if cpp_path is not None else _default_cpp_source_path()
    if not cpp_path.exists():
        raise FileNotFoundError(f"HMM C++ source not found: {cpp_path}")

    cache_dir = Path(os.environ.get("TEMA_CACHE_DIR", str(Path.home() / ".cache" / "tema")))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fall back to a temp dir if the cache dir is not writable.
        cache_dir = Path(tempfile.mkdtemp(prefix="tema-cache-"))

    so_path = cache_dir / "hmm_regime.so"
    needs_build = (not so_path.exists()) or (cpp_path.stat().st_mtime > so_path.stat().st_mtime)
    if not needs_build:
        return so_path

    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(cpp_path),
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True)
    return so_path


@dataclass
class HmmForwardResult:
    train_probs: np.ndarray
    test_probs: np.ndarray
    means: np.ndarray
    variances: np.ndarray


class CppHmmEngine:
    """ctypes wrapper around the C++ hmm_regime implementation."""

    def __init__(self, so_path: Path):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn_probs = self.lib.fit_hmm_forward_probs_1d
        self.fn_probs.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.fn_probs.restype = ctypes.c_int

    def fit_forward_probs(
        self,
        *,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
        n_states: int,
        n_iter: int,
        var_floor: float,
        trans_sticky: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_arr = np.ascontiguousarray(train_returns, dtype=np.float64)
        test_arr = np.ascontiguousarray(test_returns, dtype=np.float64)

        train_probs = np.zeros((train_arr.shape[0], n_states), dtype=np.float64)
        test_probs = np.zeros((test_arr.shape[0], n_states), dtype=np.float64)
        means = np.zeros(n_states, dtype=np.float64)
        variances = np.zeros(n_states, dtype=np.float64)

        rc = self.fn_probs(
            train_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(train_arr.shape[0]),
            test_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(test_arr.shape[0]),
            int(n_states),
            int(n_iter),
            float(var_floor),
            float(trans_sticky),
            train_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            test_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            variances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"C++ HMM forward probs failed with code {rc}")

        return train_probs, test_probs, means, variances


class PythonHmmEngine:
    """Best-effort Python fallback using hmmlearn.

    This is not guaranteed to match the C++ implementation; it exists so the code can
    run in environments without a compiler.
    """

    @staticmethod
    def _forward_filter_probs_1d(
        *,
        x: np.ndarray,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        n = int(startprob.shape[0])
        x1 = x.reshape(-1)
        mu = means.reshape(-1)
        var = np.maximum(variances.reshape(-1), 1e-12)
        norm = np.sqrt(2.0 * np.pi * var)

        # Emission probabilities [T, n]
        emis = np.exp(-0.5 * ((x1[:, None] - mu[None, :]) ** 2) / var[None, :]) / norm[None, :]
        emis = np.maximum(emis, 1e-300)

        probs = np.zeros((x1.shape[0], n), dtype=float)
        alpha = np.asarray(startprob, dtype=float) * emis[0]
        s0 = float(alpha.sum())
        alpha = (np.full(n, 1.0 / n) if s0 <= 0 or not np.isfinite(s0) else alpha / s0)
        probs[0] = alpha

        trans = np.asarray(transmat, dtype=float)
        for t in range(1, x1.shape[0]):
            alpha = (alpha @ trans) * emis[t]
            st = float(alpha.sum())
            alpha = (np.full(n, 1.0 / n) if st <= 0 or not np.isfinite(st) else alpha / st)
            probs[t] = alpha
        return probs

    def fit_forward_probs(
        self,
        *,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
        n_states: int,
        n_iter: int,
        var_floor: float,
        trans_sticky: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("hmmlearn not available") from exc

        tr = np.asarray(train_returns, dtype=float).reshape(-1, 1)
        te = np.asarray(test_returns, dtype=float).reshape(-1, 1)
        if tr.shape[0] < 10:
            # Degenerate: return uniform probabilities.
            train_probs = np.full((tr.shape[0], n_states), 1.0 / max(n_states, 1), dtype=float)
            test_probs = np.full((te.shape[0], n_states), 1.0 / max(n_states, 1), dtype=float)
            means = np.zeros(n_states, dtype=float)
            variances = np.full(n_states, max(var_floor, 1e-12), dtype=float)
            return train_probs, test_probs, means, variances

        model = GaussianHMM(
            n_components=int(n_states),
            covariance_type="full",
            random_state=42,
            n_iter=max(int(n_iter), 100),
        )
        model.fit(tr)

        means = np.asarray(getattr(model, "means_", np.zeros((n_states, 1))))[:, 0].astype(float)
        cov = np.asarray(getattr(model, "covars_", np.ones((n_states, 1, 1)))).astype(float)
        if cov.ndim == 3:
            variances = cov[:, 0, 0]
        elif cov.ndim == 2:
            variances = cov[:, 0]
        else:
            variances = cov.reshape(-1)
        variances = np.maximum(variances, float(var_floor))

        all_x = np.concatenate([tr.reshape(-1), te.reshape(-1)], axis=0)
        all_probs = self._forward_filter_probs_1d(
            x=all_x,
            startprob=np.asarray(model.startprob_, dtype=float),
            transmat=np.asarray(model.transmat_, dtype=float),
            means=means,
            variances=variances,
        )
        train_probs = all_probs[: tr.shape[0]]
        test_probs = all_probs[tr.shape[0] :]
        return train_probs, test_probs, means, variances


def get_hmm_engine(*, prefer_cpp: bool = True):
    if prefer_cpp:
        try:
            so_path = compile_cpp_hmm_library()
            return CppHmmEngine(so_path)
        except Exception:
            logger.exception("Failed to build/load C++ HMM; falling back to hmmlearn")
    return PythonHmmEngine()
