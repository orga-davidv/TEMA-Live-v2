import ctypes
import itertools
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class BacktestConfig:
    data_dir: str = "../merged_d1"
    train_ratio: float = 0.60
    init_cash: float = 100_000.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    freq: str = "D"
    batch_size: int = 1000
    max_workers: Optional[int] = None
    min_trades_per_year: float = 2.0
    min_history_rows: int = 400
    max_assets: Optional[int] = None
    verify_cpp_parity: bool = False
    parity_sample_size: int = 100
    grid_validation_ratio: float = 0.25
    grid_validation_shortlist: int = 200
    grid_validation_min_rows: int = 80
    grid_overfit_penalty: float = 0.5

    # Turnover / rebalance controls (Phase 2b)
    # Minimum fractional weight change required to trigger a rebalance (e.g., 0.001 = 0.1%)
    rebalance_min_threshold: float = 0.001
    # Enable cost-aware rebalance gating: only rebalance when expected alpha > expected costs * multiplier
    cost_aware_rebalance: bool = False
    cost_aware_rebalance_multiplier: float = 1.0
    # Lookback for expected-alpha proxy in cost-aware gating
    cost_aware_alpha_lookback: int = 20
    # Penalty applied during selection/optimization: Sharpe - lambda * annualized_turnover
    turnover_penalty_lambda: float = 0.0

    # Black-Litterman parameters
    bl_tau: float = 0.05
    bl_delta: float = 2.5
    bl_omega_scale: float = 0.25
    bl_max_weight: float = 0.15

    # C++ HMM feature generator + RF classifier
    ml_enabled: bool = True
    hmm_n_states: int = 3
    hmm_n_iter: int = 30
    hmm_var_floor: float = 1e-8
    hmm_trans_sticky: float = 0.92

    rf_n_estimators: int = 300
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 40
    rf_random_state: int = 42
    ml_prob_threshold: float = 0.55
    ml_auto_threshold: bool = True
    ml_target_exposure: float = 0.40

    # ML position scalar (separate post-ML scaling)
    ml_position_scalar_method: str = "hmm_prob"
    # HMM probability scalar shaping: scalar_raw = floor + (ceiling - floor) * p_bull
    ml_hmm_scalar_floor: float = 0.30
    ml_hmm_scalar_ceiling: float = 1.50
    ml_position_scalar: float = 1.0
    ml_position_scalar_auto: bool = True
    # Annualized target vol for ML position scalar (e.g., 0.10 = 10%)
    ml_position_scalar_target_vol: float = 0.10
    # Safety cap for ML scalar to avoid excessive leverage
    ml_position_scalar_max: float = 50.0

    # Vol-target scaling (train-based)
    vol_target_enabled: bool = True
    vol_target_annual: float = 0.10
    vol_target_max_leverage: float = 12.0
    vol_target_min_leverage: float = 0.25
    # Which series to reference for realized train volatility when computing vol-target scalar.
    # Accepted values: "ml" (use ML-filtered train returns) or "bl" (use BL train returns).
    # Default changed to "bl" so global vol-target remains BL-referenced while ML gets separate scalar
    vol_target_reference: str = "bl"
    # Whether to apply the global vol-target scalar to ML-filtered returns. Default: False => ML has its own scalar
    vol_target_apply_to_ml: bool = False

    ml_grid_search_enabled: bool = True
    ml_grid_rf_n_estimators: Tuple[int, ...] = (200, 400)
    ml_grid_rf_max_depth: Tuple[int, ...] = (4, 6, 8)
    ml_grid_rf_min_samples_leaf: Tuple[int, ...] = (20, 40, 80)
    ml_grid_target_exposure: Tuple[float, ...] = (0.10, 0.15, 0.20)
    ml_grid_hmm_n_states: Tuple[int, ...] = (2, 3, 4)


def resolve_max_workers(cfg: BacktestConfig) -> int:
    cpu_total = os.cpu_count() or 4
    if cfg.max_workers is not None and cfg.max_workers > 0:
        return int(cfg.max_workers)
    # Standard: fast volle CPU, aber einen Kern für OS/UI frei lassen.
    return max(1, cpu_total - 1)


@dataclass
class GridConfig:
    ema1_periods: Sequence[int]
    ema2_periods: Sequence[int]
    ema3_periods: Sequence[int]


class CppSignalEngine:
    def __init__(self, so_path: Path):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn = self.lib.build_signals_batch
        self.fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.fn.restype = None

    def build_signals_batch(
        self,
        ema_values: np.ndarray,
        idx1: np.ndarray,
        idx2: np.ndarray,
        idx3: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_periods, n_rows = ema_values.shape
        n_combos = idx1.shape[0]

        entries = np.zeros((n_rows, n_combos), dtype=np.uint8)
        exits = np.zeros((n_rows, n_combos), dtype=np.uint8)

        self.fn(
            ema_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_periods,
            n_rows,
            idx1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            idx2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            idx3.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_combos,
            entries.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            exits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        return entries.astype(bool), exits.astype(bool)


class CppHmmEngine:
    def __init__(self, so_path: Path):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn = self.lib.fit_predict_hmm_1d
        self.fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.fn.restype = ctypes.c_int

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

    def fit_predict(
        self,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
        n_states: int,
        n_iter: int,
        var_floor: float,
        trans_sticky: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_arr = np.ascontiguousarray(train_returns, dtype=np.float64)
        test_arr = np.ascontiguousarray(test_returns, dtype=np.float64)

        train_states = np.zeros(train_arr.shape[0], dtype=np.int32)
        test_states = np.zeros(test_arr.shape[0], dtype=np.int32)
        means = np.zeros(n_states, dtype=np.float64)
        variances = np.zeros(n_states, dtype=np.float64)

        rc = self.fn(
            train_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(train_arr.shape[0]),
            test_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(test_arr.shape[0]),
            int(n_states),
            int(n_iter),
            float(var_floor),
            float(trans_sticky),
            train_states.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            test_states.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            variances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        if rc != 0:
            raise RuntimeError(f"C++ HMM fit failed with code {rc}")

        return train_states, test_states, means, variances

    def fit_forward_probs(
        self,
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


def compile_cpp_signal_library(cpp_path: Path) -> Path:
    so_path = cpp_path.with_suffix(".so")
    needs_build = (not so_path.exists()) or (cpp_path.stat().st_mtime > so_path.stat().st_mtime)
    if not needs_build:
        return so_path

    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-fopenmp",
        str(cpp_path),
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True)
    return so_path


def compile_cpp_hmm_library(cpp_path: Path) -> Path:
    so_path = cpp_path.with_suffix(".so")
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


def generate_ema_combinations(grid_cfg: GridConfig) -> List[Tuple[int, int, int]]:
    combos: List[Tuple[int, int, int]] = []
    for e1 in grid_cfg.ema1_periods:
        for e2 in grid_cfg.ema2_periods:
            for e3 in grid_cfg.ema3_periods:
                if e1 < e2 and e1 < e3:
                    combos.append((e1, e2, e3))
    return combos


def ema_series(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def periods_per_year(freq: str) -> float:
    f = (freq or "D").upper()
    if f.startswith("D"):
        return 252.0
    if f.startswith("H"):
        return 252.0 * 24.0
    if f.startswith("W"):
        return 52.0
    if f.startswith("M"):
        return 12.0
    return 252.0


def simulate_batch_returns_and_stats(
    close: pd.Series,
    entries_np: np.ndarray,
    exits_np: np.ndarray,
    fee_rate: float,
    slippage_rate: float,
    freq: str,
) -> Dict[str, np.ndarray]:
    close_values = close.to_numpy(dtype=np.float64)
    n_rows, n_combos = entries_np.shape
    if n_rows != close_values.shape[0]:
        raise ValueError("Signal rows do not match close length")

    ret_px = np.zeros(n_rows, dtype=np.float64)
    if n_rows > 1:
        ret_px[1:] = close_values[1:] / close_values[:-1] - 1.0

    cost_rate = float(fee_rate + slippage_rate)

    pos = np.zeros(n_combos, dtype=np.int8)
    strat_rets = np.zeros((n_rows, n_combos), dtype=np.float64)
    entry_count = np.zeros(n_combos, dtype=np.int32)
    exit_count = np.zeros(n_combos, dtype=np.int32)

    # Turnover/cost tracking
    total_turnover = np.zeros(n_combos, dtype=np.float64)
    total_costs = np.zeros(n_combos, dtype=np.float64)

    trade_open = np.zeros(n_combos, dtype=bool)
    trade_ret_running = np.zeros(n_combos, dtype=np.float64)
    trade_closed_count = np.zeros(n_combos, dtype=np.int32)
    trade_win_count = np.zeros(n_combos, dtype=np.int32)
    trade_gain_sum = np.zeros(n_combos, dtype=np.float64)
    trade_loss_sum_abs = np.zeros(n_combos, dtype=np.float64)
    trade_ret_sum = np.zeros(n_combos, dtype=np.float64)

    for t in range(1, n_rows):
        strat_rets[t, :] = pos * ret_px[t]

        new_pos = pos.copy()
        ex = exits_np[t, :]
        en = entries_np[t, :]

        # If entry and exit happen on the same bar, entry wins by design.
        new_pos[ex] = 0
        new_pos[en] = 1

        turnover = np.abs(new_pos - pos)
        strat_rets[t, :] -= turnover * cost_rate

        # Track turnover and costs for diagnostics
        total_turnover += turnover
        total_costs += turnover * cost_rate

        entry_count += (new_pos > pos).astype(np.int32)
        exit_count += (new_pos < pos).astype(np.int32)

        entering = (new_pos > pos)
        staying = trade_open & (new_pos > 0)
        closing = trade_open & (new_pos == 0)

        trade_ret_running[entering] = strat_rets[t, entering]
        trade_open[entering] = True

        carry = staying & ~entering
        trade_ret_running[carry] += strat_rets[t, carry]

        trade_ret_running[closing] += strat_rets[t, closing]
        if np.any(closing):
            closed_vals = trade_ret_running[closing]
            trade_closed_count[closing] += 1
            trade_ret_sum[closing] += closed_vals

            wins = closed_vals > 0.0
            losses = closed_vals < 0.0

            idxs = np.where(closing)[0]
            if idxs.size > 0:
                win_idxs = idxs[wins]
                loss_idxs = idxs[losses]
                trade_win_count[win_idxs] += 1
                trade_gain_sum[win_idxs] += closed_vals[wins]
                trade_loss_sum_abs[loss_idxs] += np.abs(closed_vals[losses])

            trade_open[closing] = False
            trade_ret_running[closing] = 0.0

        pos = new_pos

    periods = periods_per_year(freq)
    years = max(n_rows / periods, 1e-9)

    total_return = np.prod(1.0 + strat_rets, axis=0) - 1.0
    with np.errstate(invalid="ignore"):
        annualized_return = np.where(
            (1.0 + total_return) > 0,
            (1.0 + total_return) ** (1.0 / years) - 1.0,
            np.nan,
        )

    volatility = np.std(strat_rets, axis=0) * np.sqrt(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_ratio = np.where(volatility > 0, annualized_return / volatility, np.nan)

    downside = np.where(strat_rets < 0.0, strat_rets, 0.0)
    downside_vol = np.std(downside, axis=0) * np.sqrt(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino_ratio = np.where(downside_vol > 0, annualized_return / downside_vol, np.nan)

    cum = np.cumprod(1.0 + strat_rets, axis=0)
    peak = np.maximum.accumulate(cum, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (cum - peak) / peak
    max_drawdown = np.nanmin(dd, axis=0)
    ulcer_index = np.sqrt(np.nanmean(np.square(dd), axis=0))

    gains = trade_gain_sum
    losses = trade_loss_sum_abs
    with np.errstate(divide="ignore", invalid="ignore"):
        profit_factor = np.where(losses > 0.0, gains / losses, np.inf)

    expectancy = np.divide(
        trade_ret_sum,
        trade_closed_count,
        out=np.zeros_like(trade_ret_sum, dtype=np.float64),
        where=trade_closed_count > 0,
    )
    win_rate = np.divide(
        trade_win_count,
        trade_closed_count,
        out=np.full_like(trade_ret_sum, np.nan, dtype=np.float64),
        where=trade_closed_count > 0,
    ) * 100.0

    avg_win_amount = np.zeros(n_combos, dtype=np.float64)
    avg_loss_amount = np.zeros(n_combos, dtype=np.float64)
    avg_win_amount = np.divide(
        trade_gain_sum,
        trade_win_count,
        out=np.zeros_like(trade_gain_sum, dtype=np.float64),
        where=trade_win_count > 0,
    )
    loss_trade_count = trade_closed_count - trade_win_count
    avg_loss_amount = np.divide(
        trade_loss_sum_abs,
        loss_trade_count,
        out=np.zeros_like(trade_loss_sum_abs, dtype=np.float64),
        where=loss_trade_count > 0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        payoff_ratio = np.where(avg_loss_amount > 0.0, avg_win_amount / avg_loss_amount, np.inf)

    total_trades = trade_closed_count
    trades_per_year = total_trades / years

    # Annualize turnover and costs
    with np.errstate(invalid='ignore'):
        annualized_turnover = total_turnover / years
        annual_costs = total_costs / years

    return {
        "strat_rets": strat_rets,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "ulcer_index": ulcer_index,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win_amount": avg_win_amount,
        "avg_loss_amount": avg_loss_amount,
        "payoff_ratio": payoff_ratio,
        "trades_per_year": trades_per_year,
        "annualized_turnover": annualized_turnover,
        "annual_costs": annual_costs,
    }


def precompute_ema_matrix(train_close: pd.Series, combos: Sequence[Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    all_periods = np.array(sorted({p for combo in combos for p in combo}), dtype=np.int32)
    ema_matrix = np.column_stack([ema_series(train_close, int(p)).to_numpy(dtype=np.float64) for p in all_periods])
    ema_values = np.ascontiguousarray(ema_matrix.T, dtype=np.float64)
    return all_periods, ema_values


def verify_cpp_matches_python_signals(
    train_close: pd.Series,
    all_periods: np.ndarray,
    ema_values: np.ndarray,
    combos: Sequence[Tuple[int, int, int]],
    engine: CppSignalEngine,
    sample_size: int,
) -> None:
    sample = np.array(list(combos[:sample_size]), dtype=np.int32)
    if sample.size == 0:
        return

    idx1 = np.searchsorted(all_periods, sample[:, 0]).astype(np.int32)
    idx2 = np.searchsorted(all_periods, sample[:, 1]).astype(np.int32)
    idx3 = np.searchsorted(all_periods, sample[:, 2]).astype(np.int32)
    entries_cpp, exits_cpp = engine.build_signals_batch(ema_values, idx1, idx2, idx3)

    entries_py = np.zeros_like(entries_cpp, dtype=bool)
    exits_py = np.zeros_like(exits_cpp, dtype=bool)

    for j, (e1, e2, e3) in enumerate(sample.tolist()):
        s1 = pd.Series(ema_values[np.searchsorted(all_periods, e1)], index=train_close.index)
        s2 = pd.Series(ema_values[np.searchsorted(all_periods, e2)], index=train_close.index)
        s3 = pd.Series(ema_values[np.searchsorted(all_periods, e3)], index=train_close.index)

        entries_raw = crossed_above(s1, s2) | crossed_above(s1, s3) | crossed_above(s2, s3)
        exits_raw = crossed_below(s1, s2) | crossed_below(s1, s3) | crossed_below(s2, s3)

        entries_py[:, j] = entries_raw.shift(1, fill_value=False).to_numpy(dtype=bool)
        exits_py[:, j] = exits_raw.shift(1, fill_value=False).to_numpy(dtype=bool)

    if not np.array_equal(entries_cpp, entries_py) or not np.array_equal(exits_cpp, exits_py):
        entry_diff = int(np.count_nonzero(entries_cpp != entries_py))
        exit_diff = int(np.count_nonzero(exits_cpp != exits_py))
        raise ValueError(
            f"C++ signal parity check failed (entry_diff={entry_diff}, exit_diff={exit_diff})."
        )

    print(f"C++ parity check passed for {len(sample)} combinations.")


G_CLOSE = None
G_ALL_PERIODS = None
G_EMA_VALUES = None
G_ENGINE = None
G_CFG = None


def _init_worker(
    close_values: np.ndarray,
    close_index: np.ndarray,
    all_periods: np.ndarray,
    ema_values: np.ndarray,
    so_path: str,
    cfg_dict: Dict,
) -> None:
    global G_CLOSE, G_ALL_PERIODS, G_EMA_VALUES, G_ENGINE, G_CFG
    G_CLOSE = pd.Series(close_values, index=pd.to_datetime(close_index))
    G_ALL_PERIODS = all_periods
    G_EMA_VALUES = ema_values
    G_ENGINE = CppSignalEngine(Path(so_path))
    G_CFG = BacktestConfig(**cfg_dict)


def _extract_rows_from_stats(
    stats: Dict[str, np.ndarray],
    batch_combos: np.ndarray,
    min_trades_per_year: float,
) -> List[Dict]:
    rows: List[Dict] = []

    for idx, (e1, e2, e3) in enumerate(batch_combos.tolist()):
        total_return = float(stats["total_return"][idx])
        annualized_return = float(stats["annualized_return"][idx])
        max_drawdown = float(stats["max_drawdown"][idx])
        volatility = float(stats["volatility"][idx])
        sharpe_ratio = float(stats["sharpe_ratio"][idx])
        sortino_ratio = float(stats["sortino_ratio"][idx])
        total_trades = int(stats["total_trades"][idx])
        trades_per_year = float(stats["trades_per_year"][idx])
        if trades_per_year < min_trades_per_year:
            continue

        rows.append(
            {
                "ema1_period": int(e1),
                "ema2_period": int(e2),
                "ema3_period": int(e3),
                "total_return": total_return,
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "ulcer_index": float(stats["ulcer_index"][idx]),
                "total_trades": total_trades,
                "win_rate": float(stats["win_rate"][idx]),
                "profit_factor": float(stats["profit_factor"][idx]),
                "expectancy": float(stats["expectancy"][idx]),
                "avg_win_amount": float(stats["avg_win_amount"][idx]),
                "avg_loss_amount": float(stats["avg_loss_amount"][idx]),
                "payoff_ratio": float(stats["payoff_ratio"][idx]),
                "trades_per_year": trades_per_year,
                "annualized_turnover": float(stats.get("annualized_turnover", np.zeros_like(trades_per_year))[idx]) if "annualized_turnover" in stats else float(0.0),
                "annual_costs": float(stats.get("annual_costs", np.zeros_like(trades_per_year))[idx]) if "annual_costs" in stats else float(0.0),
            }
        )
    return rows


def _process_batch(batch_combos: np.ndarray) -> List[Dict]:
    idx1 = np.searchsorted(G_ALL_PERIODS, batch_combos[:, 0]).astype(np.int32)
    idx2 = np.searchsorted(G_ALL_PERIODS, batch_combos[:, 1]).astype(np.int32)
    idx3 = np.searchsorted(G_ALL_PERIODS, batch_combos[:, 2]).astype(np.int32)

    entries_np, exits_np = G_ENGINE.build_signals_batch(G_EMA_VALUES, idx1, idx2, idx3)
    stats = simulate_batch_returns_and_stats(
        close=G_CLOSE,
        entries_np=entries_np,
        exits_np=exits_np,
        fee_rate=G_CFG.fee_rate,
        slippage_rate=G_CFG.slippage_rate,
        freq=G_CFG.freq,
    )

    return _extract_rows_from_stats(
        stats=stats,
        batch_combos=batch_combos,
        min_trades_per_year=G_CFG.min_trades_per_year,
    )


def run_parallel_grid_search_cpp(
    train_close: pd.Series,
    combos: Sequence[Tuple[int, int, int]],
    cfg: BacktestConfig,
    so_path: Path,
) -> pd.DataFrame:
    all_periods, ema_values = precompute_ema_matrix(train_close, combos)
    max_workers = resolve_max_workers(cfg)

    combo_array = np.array(combos, dtype=np.int32)
    batches = [combo_array[i : i + cfg.batch_size] for i in range(0, len(combo_array), cfg.batch_size)]

    cfg_dict = asdict(cfg)
    cfg_dict["max_workers"] = max_workers

    all_rows: List[Dict] = []
    start_ts = time.perf_counter()

    def _fmt_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(
            train_close.values.astype(np.float64),
            train_close.index.values,
            all_periods,
            ema_values,
            str(so_path),
            cfg_dict,
        ),
    ) as ex:
        futures = {ex.submit(_process_batch, batch): i + 1 for i, batch in enumerate(batches)}
        done = 0
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done += 1
            elapsed = time.perf_counter() - start_ts
            speed = done / max(elapsed, 1e-9)
            remaining = len(batches) - done
            eta = remaining / max(speed, 1e-9)
            progress = (done / len(batches)) * 100.0
            if done % 10 == 0 or done == len(batches):
                print(
                    f"Grid progress: {done}/{len(batches)} ({progress:.1f}%) | "
                    f"elapsed={_fmt_eta(elapsed)} | eta={_fmt_eta(eta)}"
                )

    if not all_rows:
        raise ValueError("No valid grid search results produced.")

    return pd.DataFrame(all_rows)


def list_asset_files(data_dir: Path, max_assets: Optional[int]) -> List[Path]:
    files = sorted(data_dir.glob("*_d1_merged.csv"))
    if max_assets is not None:
        files = files[:max_assets]
    return files


def count_csv_rows_fast(file_path: Path) -> int:
    # Schneller Vorab-Check: zählt Zeilen ohne vollständiges CSV-Parsing.
    with file_path.open("rb") as fh:
        line_count = sum(1 for _ in fh)
    return max(0, line_count - 1)  # Header abziehen


def prefilter_assets_by_min_rows(asset_files: Sequence[Path], min_rows: int) -> Tuple[List[Path], List[Tuple[str, int]]]:
    keep: List[Path] = []
    dropped: List[Tuple[str, int]] = []

    for fp in asset_files:
        try:
            rows = count_csv_rows_fast(fp)
        except Exception:
            rows = 0

        if rows >= min_rows:
            keep.append(fp)
        else:
            dropped.append((parse_asset_name(fp), rows))

    return keep, dropped


def parse_asset_name(file_path: Path) -> str:
    return file_path.name.replace("_d1_merged.csv", "")


def load_close_series_from_csv(file_path: Path) -> pd.Series:
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("CSV is empty")

    close_candidates = ["close_mid", "close", "Close", "close_bid", "close_ask"]
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if close_col is None:
        raise KeyError("No close-like column found")

    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    else:
        raise KeyError("Neither datetime nor timestamp column found")

    s = pd.Series(df[close_col].astype(float).values, index=idx, name=parse_asset_name(file_path))
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    if s.empty:
        raise ValueError("No valid close values after cleaning")
    return s


def split_train_test(close: pd.Series, train_ratio: float) -> Tuple[pd.Series, pd.Series]:
    split_idx = int(len(close) * train_ratio)
    split_idx = max(2, min(split_idx, len(close) - 2))
    train_close = close.iloc[:split_idx].copy()
    test_close = close.iloc[split_idx:].copy()
    return train_close, test_close


def build_strategy_returns_for_combo(
    close: pd.Series,
    combo: Tuple[int, int, int],
    cfg: BacktestConfig,
    so_path: Path,
) -> pd.Series:
    all_periods = np.array(sorted(combo), dtype=np.int32)
    ema_matrix = np.column_stack([ema_series(close, int(p)).to_numpy(dtype=np.float64) for p in all_periods])
    ema_values = np.ascontiguousarray(ema_matrix.T, dtype=np.float64)

    idx_map = {int(p): i for i, p in enumerate(all_periods.tolist())}
    idx1 = np.array([idx_map[int(combo[0])]], dtype=np.int32)
    idx2 = np.array([idx_map[int(combo[1])]], dtype=np.int32)
    idx3 = np.array([idx_map[int(combo[2])]], dtype=np.int32)

    engine = CppSignalEngine(so_path)
    entries_np, exits_np = engine.build_signals_batch(ema_values, idx1, idx2, idx3)

    stats = simulate_batch_returns_and_stats(
        close=close,
        entries_np=entries_np,
        exits_np=exits_np,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        freq=cfg.freq,
    )
    ret_series = pd.Series(stats["strat_rets"][:, 0], index=close.index, dtype=float).fillna(0.0)
    return ret_series, stats


def choose_best_combo_with_validation(
    subtrain_close: pd.Series,
    val_close: pd.Series,
    combos: Sequence[Tuple[int, int, int]],
    cfg: BacktestConfig,
    so_path: Path,
) -> Tuple[Tuple[int, int, int], Dict[str, float]]:
    subtrain_df = run_parallel_grid_search_cpp(subtrain_close, combos, cfg, so_path)
    shortlist_n = max(10, int(cfg.grid_validation_shortlist))
    shortlist_df = subtrain_df.nlargest(min(shortlist_n, len(subtrain_df)), "sharpe_ratio").copy()

    shortlist_combos = [
        (int(r["ema1_period"]), int(r["ema2_period"]), int(r["ema3_period"]))
        for _, r in shortlist_df.iterrows()
    ]
    val_df = run_parallel_grid_search_cpp(val_close, shortlist_combos, cfg, so_path)

    merged = shortlist_df.merge(
        val_df,
        on=["ema1_period", "ema2_period", "ema3_period"],
        suffixes=("_subtrain", "_val"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping shortlist results between subtrain and validation")

    gap = np.abs(merged["sharpe_ratio_subtrain"] - merged["sharpe_ratio_val"])
    # Apply optional turnover penalty to the selection score (conservative default: 0.0 => no penalty)
    turnover_penalty = float(getattr(cfg, "turnover_penalty_lambda", 0.0))
    # merged may contain 'annualized_turnover_val' from the validation stats; fall back to 0.0 if missing
    annual_turnover_val = merged.get("annualized_turnover_val") if "annualized_turnover_val" in merged.columns else 0.0
    # Cost-aware rebalance gate: require expected annual return > expected annual costs * multiplier
    if bool(getattr(cfg, "cost_aware_rebalance", False)) and "annual_costs_val" in merged.columns:
        multiplier = float(getattr(cfg, "cost_aware_rebalance_multiplier", 1.0))
        expected_return = merged.get("annualized_return_val") if "annualized_return_val" in merged.columns else 0.0
        expected_costs = merged.get("annual_costs_val")
        # Create a boolean mask where requirement holds
        cost_gate = np.ones(len(merged), dtype=bool)
        try:
            cost_gate = expected_return > (expected_costs * multiplier)
        except Exception:
            cost_gate = np.ones(len(merged), dtype=bool)
        # Penalize combos that fail the gate by setting selection_score very low
        merged["selection_score"] = merged["sharpe_ratio_val"] - float(cfg.grid_overfit_penalty) * gap - turnover_penalty * annual_turnover_val
        merged.loc[~cost_gate, "selection_score"] = -1e9
    else:
        merged["selection_score"] = merged["sharpe_ratio_val"] - float(cfg.grid_overfit_penalty) * gap - turnover_penalty * annual_turnover_val
    best = merged.sort_values(["selection_score", "sharpe_ratio_val"], ascending=[False, False]).iloc[0]

    combo = (int(best["ema1_period"]), int(best["ema2_period"]), int(best["ema3_period"]))
    info = {
        "subtrain_sharpe": float(best["sharpe_ratio_subtrain"]),
        "val_sharpe": float(best["sharpe_ratio_val"]),
        "selection_score": float(best["selection_score"]),
    }
    return combo, info


def annualize_return_from_series(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    cum = float((1.0 + returns).prod())
    n = len(returns)
    if n <= 1 or cum <= 0:
        return np.nan
    return float(cum ** (252.0 / n) - 1.0)


def build_black_litterman_weights(
    train_returns_df: pd.DataFrame,
    view_returns: pd.Series,
    cfg: BacktestConfig,
) -> pd.Series:
    assets = [a for a in train_returns_df.columns if a in view_returns.index]
    if not assets:
        raise ValueError("No overlapping assets between train return matrix and views")

    rets = train_returns_df[assets].fillna(0.0)
    q = view_returns[assets].astype(float).to_numpy()

    sigma = rets.cov().to_numpy() * 252.0
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    jitter = 1e-8 * np.eye(len(assets))
    sigma = sigma + jitter

    w_mkt = np.repeat(1.0 / len(assets), len(assets))
    pi = cfg.bl_delta * (sigma @ w_mkt)

    p = np.eye(len(assets))
    tau_sigma = cfg.bl_tau * sigma

    omega_diag = np.diag(p @ tau_sigma @ p.T).copy()
    omega_diag = np.where(omega_diag <= 1e-12, 1e-12, omega_diag)
    omega = np.diag(omega_diag * cfg.bl_omega_scale)

    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega)

    middle = np.linalg.pinv(inv_tau_sigma + p.T @ inv_omega @ p)
    mu_bl = middle @ (inv_tau_sigma @ pi + p.T @ inv_omega @ q)

    raw_w = np.linalg.pinv(cfg.bl_delta * sigma) @ mu_bl

    # Long-only + Weight-Cap Projektion
    raw_w = np.maximum(raw_w, 0.0)

    cap = float(cfg.bl_max_weight)
    cap = min(max(cap, 1e-6), 1.0)
    n = len(assets)
    if cap * n < 1.0:
        cap = 1.0 / n

    w_sum = float(raw_w.sum())
    if w_sum <= 0:
        raw_w = np.repeat(1.0 / n, n)
    else:
        raw_w = raw_w / w_sum

    for _ in range(20):
        clipped = np.minimum(raw_w, cap)
        deficit = 1.0 - float(clipped.sum())
        if deficit <= 1e-12:
            raw_w = clipped
            break
        free = clipped < (cap - 1e-12)
        free_sum = float(clipped[free].sum())
        if not np.any(free):
            raw_w = clipped
            break
        if free_sum <= 1e-12:
            clipped[free] += deficit / float(np.count_nonzero(free))
            raw_w = np.minimum(clipped, cap)
            continue
        clipped[free] += deficit * (clipped[free] / free_sum)
        raw_w = clipped

    raw_w = np.maximum(raw_w, 0.0)
    raw_w = raw_w / max(float(raw_w.sum()), 1e-12)

    return pd.Series(raw_w, index=assets, name="bl_weight")


def evaluate_weighted_portfolio(
    returns_df: pd.DataFrame,
    weights: pd.Series,
) -> Tuple[pd.Series, Dict[str, float]]:
    aligned = returns_df.reindex(columns=weights.index).fillna(0.0)
    portfolio_returns = aligned.mul(weights, axis=1).sum(axis=1)

    if portfolio_returns.empty:
        metrics = {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
        }
        return portfolio_returns, metrics

    cum = (1.0 + portfolio_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    total_return = float(cum.iloc[-1] - 1.0)
    ann_return = annualize_return_from_series(portfolio_returns)
    ann_vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 and np.isfinite(ann_return) else np.nan
    max_dd = float(dd.min()) if not dd.empty else np.nan

    metrics = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }
    return portfolio_returns, metrics


def compute_and_apply_vol_target(
    cfg: BacktestConfig,
    train_port_rets: pd.Series,
    test_port_rets: pd.Series,
    ml_train_rets: pd.Series,
    ml_test_rets: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, object]]:
    """
    Compute a scalar from realized train volatility and apply to both train and test portfolios.

    Reference selection configurable via cfg.vol_target_reference ("ml" or "bl").
    Returns scaled (train_port_rets, test_port_rets, ml_train_rets, ml_test_rets) and diagnostics dict.
    """
    diag: Dict[str, object] = {}
    if not bool(getattr(cfg, "vol_target_enabled", False)):
        diag["enabled"] = False
        return train_port_rets, test_port_rets, ml_train_rets, ml_test_rets, diag

    # Decide which train series to use as the reference for realized volatility
    ref_raw = getattr(cfg, "vol_target_reference", "ml")
    ref = str(ref_raw).lower() if ref_raw is not None else "ml"
    fallback_used = False

    if ref == "ml":
        # Prefer ML-filtered train returns when requested and available
        if ml_train_rets is not None and not ml_train_rets.empty:
            reference_series = ml_train_rets
        else:
            reference_series = train_port_rets
            fallback_used = True
    elif ref == "bl":
        reference_series = train_port_rets
    else:
        # Invalid config value: fallback to BL and mark fallback
        reference_series = train_port_rets
        fallback_used = True

    # Realized train volatility (annualized) using 252 trading days
    reference_train_vol = float(reference_series.std(ddof=0) * np.sqrt(252.0)) if not reference_series.empty else float(0.0)
    target_vol = float(cfg.vol_target_annual)

    unclipped_scalar = None
    clipped = False

    if reference_train_vol <= 1e-12 or not np.isfinite(reference_train_vol):
        # Degenerate case: no variation in reference train returns. Use safety cap and mark clipping.
        unclipped_scalar = float(cfg.vol_target_max_leverage)
        scalar = float(cfg.vol_target_max_leverage)
        clipped = True
    else:
        unclipped_scalar = float(target_vol / reference_train_vol)
        scalar = float(np.clip(unclipped_scalar, cfg.vol_target_min_leverage, cfg.vol_target_max_leverage))
        clipped = not np.isclose(unclipped_scalar, scalar)

    # Apply scalar to BL train and test portfolio returns always
    scaled_train = train_port_rets * scalar
    scaled_test = test_port_rets * scalar

    # Decide whether to apply the global vol-target scalar to ML-filtered returns
    apply_to_ml = bool(getattr(cfg, "vol_target_apply_to_ml", False))
    if ml_train_rets is not None and not ml_train_rets.empty:
        if apply_to_ml:
            scaled_ml_train = ml_train_rets * scalar
            scaled_ml_test = ml_test_rets * scalar
        else:
            # Keep ML returns as-is (ML-specific scalar was applied earlier)
            scaled_ml_train = ml_train_rets
            scaled_ml_test = ml_test_rets
    else:
        scaled_ml_train = ml_train_rets
        scaled_ml_test = ml_test_rets

    diag = {
        "enabled": True,
        "reference": ref,
        "reference_train_vol": float(reference_train_vol),
        "fallback_used": bool(fallback_used),
        "scalar": float(scalar),
        "unclipped_scalar": float(unclipped_scalar),
        "target_vol": float(target_vol),
        "min_leverage": float(cfg.vol_target_min_leverage),
        "max_leverage": float(cfg.vol_target_max_leverage),
        "clipped": bool(bool(clipped)),
        "apply_to_ml": bool(apply_to_ml),
    }

    return scaled_train, scaled_test, scaled_ml_train, scaled_ml_test, diag


def compute_and_apply_ml_position_scalar(
    cfg: BacktestConfig,
    ml_train_rets: pd.Series,
    ml_test_rets: pd.Series,
    train_port_rets: Optional[pd.Series] = None,
    test_port_rets: Optional[pd.Series] = None,
    hmm_engine: Optional[CppHmmEngine] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    """
    Compute a post-ML position scalar and apply it only to ML train/test returns.

    Default behavior: use HMM bull-state forward-filtered probability as a dynamic, time-varying
    raw scalar (Variant C). Calibrate an overall factor on the ML *train* set so that the
    resulting ML-scaled train volatility matches cfg.ml_position_scalar_target_vol.

    Fallbacks:
    - If HMM inputs or engine are missing, fall back to the original simple vol-based scalar
      behaviour (keeps backwards compatibility).

    Returns (scaled_ml_train, scaled_ml_test, diagnostics)
    """
    diag: Dict[str, object] = {}

    # If ML returns are missing or empty, do nothing
    if ml_train_rets is None or ml_train_rets.empty:
        diag["enabled"] = False
        diag["auto"] = bool(getattr(cfg, "ml_position_scalar_auto", True))
        return ml_train_rets, ml_test_rets, diag

    target_vol = float(getattr(cfg, "ml_position_scalar_target_vol", 0.10))
    max_scalar = float(getattr(cfg, "ml_position_scalar_max", 50.0))
    hmm_floor = float(getattr(cfg, "ml_hmm_scalar_floor", 0.30))
    hmm_ceiling = float(getattr(cfg, "ml_hmm_scalar_ceiling", 1.50))

    method = getattr(cfg, "ml_position_scalar_method", "hmm_prob")

    # Helper: annualized vol
    def _ann_vol(series: pd.Series) -> float:
        return float(series.std(ddof=0) * np.sqrt(252.0)) if (series is not None and not series.empty) else 0.0

    def _shape_hmm_raw(raw: pd.Series) -> pd.Series:
        lo = min(hmm_floor, hmm_ceiling)
        hi = max(hmm_floor, hmm_ceiling)
        clipped = raw.clip(lower=0.0, upper=1.0)
        return (lo + (hi - lo) * clipped).astype(float)

    # Try HMM-probability scalar when requested and feasible
    if method == "hmm_prob" and train_port_rets is not None and test_port_rets is not None:
        n_states = int(getattr(cfg, "hmm_n_states", 3))

        def _calibrate_from_raw(
            raw_train: pd.Series,
            raw_test: pd.Series,
            method_name: str,
            bull_state: int,
        ) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
            pre_vol = _ann_vol(ml_train_rets * raw_train)
            if pre_vol <= 0 or not np.isfinite(pre_vol):
                factor = 0.0
            else:
                factor = target_vol / pre_vol

            scalar_train_cal = (raw_train * factor).clip(upper=max_scalar)
            scalar_test = (raw_test * factor).clip(upper=max_scalar)
            scaled_ml_train = ml_train_rets * scalar_train_cal
            scaled_ml_test = ml_test_rets * scalar_test

            out_diag: Dict[str, object] = {
                "enabled": True,
                "method": method_name,
                "hmm_n_states": n_states,
                "bull_state": int(bull_state),
                "factor": float(factor),
                "pre_vol": float(pre_vol),
                "post_vol": float(_ann_vol(scaled_ml_train)),
                "train_raw_scalar_stats": {
                    "mean": float(raw_train.mean()),
                    "median": float(raw_train.median()),
                    "p90": float(np.percentile(raw_train, 90)) if len(raw_train) > 0 else np.nan,
                    "max": float(raw_train.max()),
                },
                "test_raw_scalar_stats": {
                    "mean": float(raw_test.mean()),
                    "median": float(raw_test.median()),
                    "p90": float(np.percentile(raw_test, 90)) if len(raw_test) > 0 else np.nan,
                    "max": float(raw_test.max()),
                },
                "max_scalar": max_scalar,
                "hmm_scalar_floor": hmm_floor,
                "hmm_scalar_ceiling": hmm_ceiling,
                "applied_scalar_train_mean": float(scalar_train_cal.mean()),
            }
            return scaled_ml_train, scaled_ml_test, out_diag

        # Primary path: Python GaussianHMM forward-filtered probabilities (Variant C semantics)
        try:
            from hmmlearn.hmm import GaussianHMM

            train_series = train_port_rets.astype(float).fillna(0.0)
            test_series = test_port_rets.astype(float).fillna(0.0)
            train_X = train_series.to_numpy(dtype=np.float64).reshape(-1, 1)
            if train_X.shape[0] < 10:
                raise ValueError("Not enough samples for python HMM scalar")

            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                random_state=42,
                n_iter=max(int(getattr(cfg, "hmm_n_iter", 30)), 100),
            )
            model.fit(train_X)
            means = np.asarray(model.means_[:, 0], dtype=float)
            bull_state = int(np.argmax(means)) if means.size > 0 else 0

            all_series = pd.concat([train_series, test_series])
            all_series = all_series[~all_series.index.duplicated(keep="last")]

            def _forward_filter_probs_1d(hmm_model: GaussianHMM, x: np.ndarray) -> np.ndarray:
                n = hmm_model.n_components
                start = np.asarray(hmm_model.startprob_, dtype=float)
                trans = np.asarray(hmm_model.transmat_, dtype=float)
                mu = np.asarray(hmm_model.means_[:, 0], dtype=float)

                cov = np.asarray(hmm_model.covars_, dtype=float)
                if cov.ndim == 3:
                    vars_ = cov[:, 0, 0]
                elif cov.ndim == 2:
                    vars_ = cov[:, 0]
                else:
                    vars_ = cov.reshape(-1)
                vars_ = np.maximum(vars_, 1e-12)

                x2 = x.reshape(-1, 1)
                norm = np.sqrt(2.0 * np.pi * vars_)
                emis = np.exp(-0.5 * ((x2 - mu) ** 2) / vars_) / norm
                emis = np.maximum(emis, 1e-300)

                probs = np.zeros((x2.shape[0], n), dtype=float)
                alpha = start * emis[0]
                s0 = alpha.sum()
                alpha = np.full(n, 1.0 / n, dtype=float) if s0 <= 0 or not np.isfinite(s0) else alpha / s0
                probs[0] = alpha
                for t in range(1, x2.shape[0]):
                    alpha = (alpha @ trans) * emis[t]
                    st = alpha.sum()
                    alpha = np.full(n, 1.0 / n, dtype=float) if st <= 0 or not np.isfinite(st) else alpha / st
                    probs[t] = alpha
                return probs

            all_X = all_series.to_numpy(dtype=np.float64).reshape(-1, 1)
            all_probs = _forward_filter_probs_1d(model, all_X)
            p_bull_all = pd.Series(all_probs[:, bull_state], index=all_series.index).shift(1)
            raw_train = p_bull_all.reindex(ml_train_rets.index).fillna(0.0)
            raw_test = p_bull_all.reindex(ml_test_rets.index).fillna(0.0)

            if float(raw_test.abs().sum()) <= 1e-12:
                raise ValueError("Python HMM scalar degenerated: test segment all zeros")

            return _calibrate_from_raw(
                _shape_hmm_raw(raw_train),
                _shape_hmm_raw(raw_test),
                "hmm_prob_python",
                bull_state,
            )
        except Exception as exc:
            diag["python_hmm_error"] = str(exc)

        # Secondary path: C++ HMM forward probabilities
        if hmm_engine is not None:
            try:
                tr = train_port_rets.fillna(0.0).to_numpy(dtype=np.float64)
                te = test_port_rets.fillna(0.0).to_numpy(dtype=np.float64)

                train_probs, test_probs, means, variances = hmm_engine.fit_forward_probs(
                    train_returns=tr,
                    test_returns=te,
                    n_states=n_states,
                    n_iter=int(getattr(cfg, "hmm_n_iter", 30)),
                    var_floor=float(getattr(cfg, "hmm_var_floor", 1e-8)),
                    trans_sticky=float(getattr(cfg, "hmm_trans_sticky", 0.92)),
                )
                bull_state = int(np.argmax(means)) if means.size > 0 else 0
                raw_train = (
                    pd.Series(train_probs[:, bull_state], index=train_port_rets.index)
                    .shift(1)
                    .reindex(ml_train_rets.index)
                    .fillna(0.0)
                )
                raw_test = (
                    pd.Series(test_probs[:, bull_state], index=test_port_rets.index)
                    .shift(1)
                    .reindex(ml_test_rets.index)
                    .fillna(0.0)
                )

                if float(raw_test.abs().sum()) <= 1e-12:
                    raise ValueError("C++ HMM scalar degenerated: test segment all zeros")

                return _calibrate_from_raw(
                    _shape_hmm_raw(raw_train),
                    _shape_hmm_raw(raw_test),
                    "hmm_prob_cpp",
                    bull_state,
                )
            except Exception as exc:
                # If anything fails, fall back to original vol-based scalar but record the error
                diag["hmm_error"] = str(exc)

    # Fallback/compatibility behaviour: original vol-based scalar
    train_ml_vol = float(ml_train_rets.std(ddof=0) * np.sqrt(252.0))
    unclipped_scalar = None
    clipped = False

    if bool(getattr(cfg, "ml_position_scalar_auto", True)):
        # Auto compute scalar from realized train ML volatility
        if train_ml_vol <= 1e-12 or not np.isfinite(train_ml_vol):
            unclipped_scalar = max_scalar
            scalar = max_scalar
            clipped = True
        else:
            unclipped_scalar = float(target_vol / train_ml_vol)
            scalar = float(np.clip(unclipped_scalar, 0.0, max_scalar))
            clipped = not np.isclose(unclipped_scalar, scalar)
    else:
        unclipped_scalar = float(getattr(cfg, "ml_position_scalar", 1.0))
        scalar = float(min(unclipped_scalar, max_scalar))
        clipped = not np.isclose(unclipped_scalar, scalar)

    scaled_ml_train = ml_train_rets * scalar
    scaled_ml_test = ml_test_rets * scalar

    diag = {
        "enabled": True,
        "method": "vol_fallback",
        "auto": bool(getattr(cfg, "ml_position_scalar_auto", True)),
        "scalar": float(scalar),
        "unclipped_scalar": float(unclipped_scalar),
        "train_ml_vol": float(train_ml_vol),
        "target_vol": float(target_vol),
        "max_scalar": max_scalar,
        "clipped": bool(clipped),
    }

    return scaled_ml_train, scaled_ml_test, diag


def save_equity_curve_chart(
    series_map: Dict[str, pd.Series],
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(12, 6))
    for label, rets in series_map.items():
        s = rets.fillna(0.0).astype(float)
        eq = (1.0 + s).cumprod()
        plt.plot(eq.index, eq.values, label=label, linewidth=1.4)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (Start = 1.0)")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def build_rf_feature_matrix(
    returns: pd.Series,
    hmm_probs: np.ndarray,
) -> pd.DataFrame:
    r = returns.fillna(0.0)
    f = pd.DataFrame(index=r.index)
    f["ret_1"] = r.shift(1).fillna(0.0)
    f["ret_5"] = r.rolling(5, min_periods=1).mean().shift(1).fillna(0.0)
    f["ret_20"] = r.rolling(20, min_periods=1).mean().shift(1).fillna(0.0)
    f["vol_20"] = r.rolling(20, min_periods=2).std(ddof=0).shift(1).fillna(0.0)
    f["abs_ret_1"] = r.abs().shift(1).fillna(0.0)

    for k in range(hmm_probs.shape[1]):
        f[f"hmm_p_{k}"] = hmm_probs[:, k]

    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def calibrate_soft_threshold_for_target_exposure(
    probs: np.ndarray,
    target_exposure: float,
    fallback_threshold: float,
) -> float:
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return float(np.clip(fallback_threshold, 0.01, 0.99))

    target = float(np.clip(target_exposure, 1e-4, 0.999))
    fallback = float(np.clip(fallback_threshold, 0.01, 0.99))

    def mean_scale(threshold: float) -> float:
        denom = max(1.0 - threshold, 1e-9)
        scale = np.clip((p - threshold) / denom, 0.0, 1.0)
        return float(np.mean(scale))

    # Exposure monoton fallend in threshold; robuste Bisektion auf [0.01, 0.99].
    lo, hi = 0.01, 0.99
    lo_exp = mean_scale(lo)
    hi_exp = mean_scale(hi)

    if target >= lo_exp:
        return lo
    if target <= hi_exp:
        return hi

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        mid_exp = mean_scale(mid)
        if mid_exp > target:
            lo = mid
        else:
            hi = mid

    thr = 0.5 * (lo + hi)
    if not np.isfinite(thr):
        return fallback
    return float(np.clip(thr, 0.01, 0.99))


def apply_turnover_reduction_gates(
    raw_scale: np.ndarray,
    probs: np.ndarray,
    returns: pd.Series,
    cfg: BacktestConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply turnover reduction directly on exposure path:
    1) skip tiny exposure deltas (min rebalance threshold),
    2) optionally require expected alpha to exceed expected cost for risk-increasing changes.
    """
    scale = np.asarray(raw_scale, dtype=np.float64).copy()
    p = np.asarray(probs, dtype=np.float64).copy()
    if scale.size == 0:
        return scale, {
            "threshold_skips": 0.0,
            "cost_skips": 0.0,
            "annualized_turnover_raw": 0.0,
            "annualized_turnover_gated": 0.0,
            "annualized_cost_drag_raw": 0.0,
            "annualized_cost_drag_gated": 0.0,
        }

    n = min(scale.size, p.size, len(returns))
    scale = np.clip(scale[:n], 0.0, 1.0)
    p = np.clip(p[:n], 0.0, 1.0)
    r = returns.iloc[:n].fillna(0.0)

    min_threshold = max(float(getattr(cfg, "rebalance_min_threshold", 0.0)), 0.0)
    cost_aware = bool(getattr(cfg, "cost_aware_rebalance", False))
    cost_mult = max(float(getattr(cfg, "cost_aware_rebalance_multiplier", 1.0)), 0.0)
    lookback = max(int(getattr(cfg, "cost_aware_alpha_lookback", 20)), 2)
    cost_per_unit = float(cfg.fee_rate + cfg.slippage_rate)

    alpha_base = (
        r.abs()
        .rolling(lookback, min_periods=max(2, lookback // 2))
        .mean()
        .shift(1)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    signal_strength = np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)
    expected_alpha = signal_strength * alpha_base

    gated = scale.copy()
    threshold_skips = 0
    cost_skips = 0
    for i in range(1, n):
        prev = gated[i - 1]
        proposed = gated[i]
        delta = proposed - prev

        if np.abs(delta) < min_threshold:
            gated[i] = prev
            threshold_skips += 1
            continue

        # Apply cost-aware gate only when increasing risk; de-risking is always allowed.
        if cost_aware and delta > 0.0:
            exp_cost = np.abs(delta) * cost_per_unit
            exp_alpha = expected_alpha[i] if np.isfinite(expected_alpha[i]) else 0.0
            if exp_alpha <= (exp_cost * cost_mult):
                gated[i] = prev
                cost_skips += 1

    ppy = periods_per_year(cfg.freq)
    turnover_raw = np.abs(np.diff(scale, prepend=0.0))
    turnover_gated = np.abs(np.diff(gated, prepend=0.0))
    annual_turnover_raw = float(np.mean(turnover_raw) * ppy)
    annual_turnover_gated = float(np.mean(turnover_gated) * ppy)

    diag = {
        "threshold_skips": float(threshold_skips),
        "cost_skips": float(cost_skips),
        "threshold_skip_ratio": float(threshold_skips / max(1, n - 1)),
        "cost_skip_ratio": float(cost_skips / max(1, n - 1)),
        "annualized_turnover_raw": annual_turnover_raw,
        "annualized_turnover_gated": annual_turnover_gated,
        "annualized_cost_drag_raw": float(annual_turnover_raw * cost_per_unit),
        "annualized_cost_drag_gated": float(annual_turnover_gated * cost_per_unit),
        "mean_order_size_raw": float(np.mean(turnover_raw)),
        "mean_order_size_gated": float(np.mean(turnover_gated)),
        "median_order_size_raw": float(np.median(turnover_raw)),
        "median_order_size_gated": float(np.median(turnover_gated)),
        "p90_order_size_raw": float(np.percentile(turnover_raw, 90)),
        "p90_order_size_gated": float(np.percentile(turnover_gated, 90)),
    }
    return gated, diag


def apply_hmm_softprob_rf_strategy(
    train_port_rets: pd.Series,
    test_port_rets: pd.Series,
    cfg: BacktestConfig,
    hmm_engine: CppHmmEngine,
) -> Tuple[pd.Series, pd.Series, Dict[str, float], pd.DataFrame]:
    tr = train_port_rets.fillna(0.0)
    te = test_port_rets.fillna(0.0)

    train_probs, test_probs, means, variances = hmm_engine.fit_forward_probs(
        train_returns=tr.to_numpy(dtype=np.float64),
        test_returns=te.to_numpy(dtype=np.float64),
        n_states=cfg.hmm_n_states,
        n_iter=cfg.hmm_n_iter,
        var_floor=cfg.hmm_var_floor,
        trans_sticky=cfg.hmm_trans_sticky,
    )

    x_train = build_rf_feature_matrix(tr, train_probs)
    x_test = build_rf_feature_matrix(te, test_probs)

    y_train = (tr.shift(-1) > 0.0).astype(int)
    x_train = x_train.iloc[:-1]
    y_train = y_train.iloc[:-1]

    clf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        random_state=cfg.rf_random_state,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    p_train = clf.predict_proba(build_rf_feature_matrix(tr, train_probs))[:, 1]
    p_test = clf.predict_proba(x_test)[:, 1]

    threshold = float(cfg.ml_prob_threshold)
    if bool(cfg.ml_auto_threshold):
        threshold = calibrate_soft_threshold_for_target_exposure(
            probs=p_train,
            target_exposure=cfg.ml_target_exposure,
            fallback_threshold=threshold,
        )

    scale_train_raw = np.clip((p_train - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0)
    scale_test_raw = np.clip((p_test - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0)

    scale_all_raw = np.concatenate([scale_train_raw, scale_test_raw])
    probs_all = np.concatenate([p_train, p_test])
    returns_all = pd.concat([tr, te], axis=0)
    scale_all_gated, gate_diag = apply_turnover_reduction_gates(
        raw_scale=scale_all_raw,
        probs=probs_all,
        returns=returns_all,
        cfg=cfg,
    )
    n_train = len(tr)
    scale_train = scale_all_gated[:n_train]
    scale_test = scale_all_gated[n_train:]

    ml_train = pd.Series(tr.to_numpy(dtype=np.float64) * scale_train, index=tr.index, name="portfolio_return_ml")
    ml_test = pd.Series(te.to_numpy(dtype=np.float64) * scale_test, index=te.index, name="portfolio_return_ml")

    diag = {
        "n_states": float(cfg.hmm_n_states),
        "rf_n_estimators": float(cfg.rf_n_estimators),
        "rf_max_depth": float(cfg.rf_max_depth),
        "rf_min_samples_leaf": float(cfg.rf_min_samples_leaf),
        "ml_prob_threshold": float(cfg.ml_prob_threshold),
        "ml_threshold_used": threshold,
        "ml_auto_threshold": float(1.0 if cfg.ml_auto_threshold else 0.0),
        "ml_target_exposure": float(cfg.ml_target_exposure),
        "train_avg_exposure": float(scale_train.mean()),
        "test_avg_exposure": float(scale_test.mean()),
        "train_avg_exposure_raw": float(np.mean(scale_train_raw)),
        "test_avg_exposure_raw": float(np.mean(scale_test_raw)),
        "rebalance_min_threshold": float(getattr(cfg, "rebalance_min_threshold", 0.0)),
        "cost_aware_rebalance": float(1.0 if bool(getattr(cfg, "cost_aware_rebalance", False)) else 0.0),
        "cost_aware_rebalance_multiplier": float(getattr(cfg, "cost_aware_rebalance_multiplier", 1.0)),
        "cost_aware_alpha_lookback": float(getattr(cfg, "cost_aware_alpha_lookback", 20)),
    }
    diag.update({f"gate_{k}": float(v) for k, v in gate_diag.items()})

    hmm_df = pd.DataFrame(
        {
            "state": np.arange(cfg.hmm_n_states, dtype=int),
            "mean_return": means,
            "variance": variances,
        }
    )
    return ml_train, ml_test, diag, hmm_df


def run_ml_parameter_grid_search(
    train_port_rets: pd.Series,
    test_port_rets: pd.Series,
    cfg: BacktestConfig,
    hmm_engine: CppHmmEngine,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    combos = list(
        itertools.product(
            cfg.ml_grid_rf_n_estimators,
            cfg.ml_grid_rf_max_depth,
            cfg.ml_grid_rf_min_samples_leaf,
            cfg.ml_grid_target_exposure,
            cfg.ml_grid_hmm_n_states,
        )
    )
    print(f"ML grid search: evaluating {len(combos)} combinations")

    for n_est, depth, min_leaf, target_exp, n_states in combos:
        local_cfg = replace(
            cfg,
            rf_n_estimators=int(n_est),
            rf_max_depth=int(depth),
            rf_min_samples_leaf=int(min_leaf),
            ml_target_exposure=float(target_exp),
            hmm_n_states=int(n_states),
            ml_auto_threshold=True,
        )

        try:
            ml_train, ml_test, diag, _ = apply_hmm_softprob_rf_strategy(
                train_port_rets=train_port_rets,
                test_port_rets=test_port_rets,
                cfg=local_cfg,
                hmm_engine=hmm_engine,
            )

            _, train_m = evaluate_weighted_portfolio(
                pd.DataFrame({"p": ml_train}),
                pd.Series({"p": 1.0}),
            )
            _, test_m = evaluate_weighted_portfolio(
                pd.DataFrame({"p": ml_test}),
                pd.Series({"p": 1.0}),
            )

            rows.append(
                {
                    "rf_n_estimators": int(n_est),
                    "rf_max_depth": int(depth),
                    "rf_min_samples_leaf": int(min_leaf),
                    "ml_target_exposure": float(target_exp),
                    "hmm_n_states": int(n_states),
                    "ml_threshold_used": float(diag.get("ml_threshold_used", np.nan)),
                    "train_avg_exposure": float(diag.get("train_avg_exposure", np.nan)),
                    "test_avg_exposure": float(diag.get("test_avg_exposure", np.nan)),
                    "train_total_return": float(train_m["total_return"]),
                    "train_annualized_return": float(train_m["annualized_return"]),
                    "train_annualized_volatility": float(train_m["annualized_volatility"]),
                    "train_sharpe_ratio": float(train_m["sharpe_ratio"]),
                    "train_max_drawdown": float(train_m["max_drawdown"]),
                    "test_total_return": float(test_m["total_return"]),
                    "test_annualized_return": float(test_m["annualized_return"]),
                    "test_annualized_volatility": float(test_m["annualized_volatility"]),
                    "test_sharpe_ratio": float(test_m["sharpe_ratio"]),
                    "test_max_drawdown": float(test_m["max_drawdown"]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "rf_n_estimators": int(n_est),
                    "rf_max_depth": int(depth),
                    "rf_min_samples_leaf": int(min_leaf),
                    "ml_target_exposure": float(target_exp),
                    "hmm_n_states": int(n_states),
                    "error": str(exc),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "test_sharpe_ratio" in df.columns:
        df = df.sort_values(
            by=["test_sharpe_ratio", "test_annualized_return"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
    return df


def stat_value_as_float(stats: Optional[Dict[str, np.ndarray]], key: str, default: float = np.nan) -> float:
    if stats is None or key not in stats:
        return float(default)
    arr = np.asarray(stats.get(key))
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def run_asset_pipeline(
    asset_file: Path,
    combos: Sequence[Tuple[int, int, int]],
    cfg: BacktestConfig,
    so_path: Path,
    parity_checked: bool,
) -> Tuple[Dict, pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    asset = parse_asset_name(asset_file)
    close = load_close_series_from_csv(asset_file)

    if len(close) < cfg.min_history_rows:
        raise ValueError(f"History too short ({len(close)} rows)")

    train_close, test_close = split_train_test(close, cfg.train_ratio)

    split_idx = int(len(train_close) * (1.0 - cfg.grid_validation_ratio))
    split_idx = max(cfg.grid_validation_min_rows, min(split_idx, len(train_close) - cfg.grid_validation_min_rows))
    subtrain_close = train_close.iloc[:split_idx].copy()
    val_close = train_close.iloc[split_idx:].copy()

    if cfg.verify_cpp_parity and not parity_checked:
        all_periods, ema_values = precompute_ema_matrix(train_close, combos)
        parity_engine = CppSignalEngine(so_path)
        verify_cpp_matches_python_signals(
            train_close=train_close,
            all_periods=all_periods,
            ema_values=ema_values,
            combos=combos,
            engine=parity_engine,
            sample_size=cfg.parity_sample_size,
        )
        parity_checked = True

    best_combo, selection_info = choose_best_combo_with_validation(
        subtrain_close=subtrain_close,
        val_close=val_close,
        combos=combos,
        cfg=cfg,
        so_path=so_path,
    )

    train_returns, train_stats = build_strategy_returns_for_combo(train_close, best_combo, cfg, so_path)
    test_returns, test_stats = build_strategy_returns_for_combo(test_close, best_combo, cfg, so_path)
    train_bh_returns = train_close.pct_change().fillna(0.0).astype(float)
    test_bh_returns = test_close.pct_change().fillna(0.0).astype(float)

    _, train_eval = evaluate_weighted_portfolio(pd.DataFrame({"p": train_returns}), pd.Series({"p": 1.0}))

    summary = {
        "asset": asset,
        "rows_total": int(len(close)),
        "rows_train": int(len(train_close)),
        "rows_test": int(len(test_close)),
        "ema1_period": best_combo[0],
        "ema2_period": best_combo[1],
        "ema3_period": best_combo[2],
        "train_sharpe": float(train_eval["sharpe_ratio"]),
        "train_total_return": float(train_eval["total_return"]),
        "train_annualized_return": float(train_eval["annualized_return"]),
        "train_max_drawdown": float(train_eval["max_drawdown"]),
        "train_trades": int(round(stat_value_as_float(train_stats, "total_trades", 0.0))),
        "train_annualized_turnover": stat_value_as_float(train_stats, "annualized_turnover", np.nan),
        "train_annual_costs": stat_value_as_float(train_stats, "annual_costs", np.nan),
        "test_trades": int(round(stat_value_as_float(test_stats, "total_trades", 0.0))),
        "test_annualized_turnover": stat_value_as_float(test_stats, "annualized_turnover", np.nan),
        "test_annual_costs": stat_value_as_float(test_stats, "annual_costs", np.nan),
        "subtrain_sharpe": float(selection_info["subtrain_sharpe"]),
        "val_sharpe": float(selection_info["val_sharpe"]),
        "selection_score": float(selection_info["selection_score"]),
        "view_q": annualize_return_from_series(train_returns),
    }

    train_returns.name = asset
    test_returns.name = asset
    train_bh_returns.name = asset
    test_bh_returns.name = asset
    return summary, train_returns, test_returns, train_bh_returns, test_bh_returns, parity_checked


def main() -> None:
    cfg = BacktestConfig()
    # Lightweight environment overrides to support isolated ablation experiments.
    # Supported env vars (set only the ones you need):
    #  REB_MIN_THRESHOLD - float -> overrides cfg.rebalance_min_threshold
    #  COST_AWARE_REBALANCE - bool-like -> overrides cfg.cost_aware_rebalance
    #  COST_AWARE_REBALANCE_MULTIPLIER - float -> overrides cfg.cost_aware_rebalance_multiplier
    #  TURNOVER_PENALTY_LAMBDA - float -> overrides cfg.turnover_penalty_lambda
    try:
        env = os.environ
        if "REB_MIN_THRESHOLD" in env:
            cfg.rebalance_min_threshold = float(env["REB_MIN_THRESHOLD"])
        if "COST_AWARE_REBALANCE" in env:
            v = str(env["COST_AWARE_REBALANCE"]).strip().lower()
            cfg.cost_aware_rebalance = v in ("1", "true", "yes", "on")
        if "COST_AWARE_REBALANCE_MULTIPLIER" in env:
            cfg.cost_aware_rebalance_multiplier = float(env["COST_AWARE_REBALANCE_MULTIPLIER"])
        if "TURNOVER_PENALTY_LAMBDA" in env:
            cfg.turnover_penalty_lambda = float(env["TURNOVER_PENALTY_LAMBDA"])
    except Exception as exc:
        print(f"Warning: failed to apply env overrides: {exc}")

    max_workers = resolve_max_workers(cfg)
    grid_cfg = GridConfig(
        ema1_periods=list(range(4, 40, 3)),
        ema2_periods=list(range(80, 200, 3)),
        ema3_periods=list(range(100, 200, 3)),
    )

    here = Path(__file__).resolve().parent
    data_dir = (here / cfg.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cpp_path = here / "grid_signals.cpp"
    so_path = compile_cpp_signal_library(cpp_path)
    hmm_cpp_path = here / "hmm_regime.cpp"
    hmm_so_path = compile_cpp_hmm_library(hmm_cpp_path)
    hmm_engine = CppHmmEngine(hmm_so_path)

    combos = generate_ema_combinations(grid_cfg)
    print(f"Total EMA combinations: {len(combos)}")
    print(f"Python workers per asset-grid: {max_workers}")

    asset_files = list_asset_files(data_dir, cfg.max_assets)
    if not asset_files:
        raise ValueError(f"No *_d1_merged.csv files found in {data_dir}")

    print(f"Assets selected (raw): {len(asset_files)}")
    asset_files, dropped_assets = prefilter_assets_by_min_rows(asset_files, cfg.min_history_rows)
    print(
        f"Assets after min-row prefilter (>= {cfg.min_history_rows}): {len(asset_files)} | "
        f"dropped: {len(dropped_assets)}"
    )
    if dropped_assets:
        preview = ", ".join(f"{a}({r})" for a, r in dropped_assets[:10])
        print(f"Dropped preview: {preview}")

    if not asset_files:
        raise ValueError("No assets left after min-row prefilter.")

    summaries: List[Dict] = []
    train_ret_list: List[pd.Series] = []
    test_ret_list: List[pd.Series] = []
    train_bh_ret_list: List[pd.Series] = []
    test_bh_ret_list: List[pd.Series] = []

    parity_checked = False
    assets_start_ts = time.perf_counter()

    def _fmt_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    for i, asset_file in enumerate(asset_files, 1):
        asset = parse_asset_name(asset_file)
        try:
            summary, train_ret, test_ret, train_bh_ret, test_bh_ret, parity_checked = run_asset_pipeline(
                asset_file=asset_file,
                combos=combos,
                cfg=cfg,
                so_path=so_path,
                parity_checked=parity_checked,
            )
            summaries.append(summary)
            train_ret_list.append(train_ret)
            test_ret_list.append(test_ret)
            train_bh_ret_list.append(train_bh_ret)
            test_bh_ret_list.append(test_bh_ret)
            status = "OK"
        except Exception as exc:
            status = f"SKIP ({exc})"

        elapsed = time.perf_counter() - assets_start_ts
        speed = i / max(elapsed, 1e-9)
        remaining = len(asset_files) - i
        eta = remaining / max(speed, 1e-9)
        progress = (i / len(asset_files)) * 100.0

        if status == "OK":
            print(
                f"Asset progress: {i}/{len(asset_files)} ({progress:.1f}%) | "
                f"{asset} {status} | best EMA=({summary['ema1_period']},{summary['ema2_period']},{summary['ema3_period']}) | "
                f"elapsed={_fmt_eta(elapsed)} | eta={_fmt_eta(eta)}"
            )
        else:
            print(
                f"Asset progress: {i}/{len(asset_files)} ({progress:.1f}%) | "
                f"{asset} {status} | elapsed={_fmt_eta(elapsed)} | eta={_fmt_eta(eta)}"
            )

    if not summaries:
        raise ValueError("No assets produced valid strategy results.")

    summary_df = pd.DataFrame(summaries).sort_values("train_sharpe", ascending=False).reset_index(drop=True)

    train_returns_df = pd.concat(train_ret_list, axis=1).sort_index().fillna(0.0)
    test_returns_df = pd.concat(test_ret_list, axis=1).sort_index().fillna(0.0)
    train_bh_returns_df = pd.concat(train_bh_ret_list, axis=1).sort_index().fillna(0.0)
    test_bh_returns_df = pd.concat(test_bh_ret_list, axis=1).sort_index().fillna(0.0)

    view_q = summary_df.set_index("asset")["view_q"]
    bl_weights = build_black_litterman_weights(train_returns_df, view_q, cfg).sort_values(ascending=False)

    train_port_rets, train_metrics = evaluate_weighted_portfolio(train_returns_df, bl_weights)
    test_port_rets, test_metrics = evaluate_weighted_portfolio(test_returns_df, bl_weights)

    ml_train_rets = train_port_rets.copy()
    ml_test_rets = test_port_rets.copy()
    ml_state_df = pd.DataFrame()
    ml_diag: Dict[str, float] = {}
    ml_train_metrics = train_metrics.copy()
    ml_test_metrics = test_metrics.copy()
    ml_grid_df = pd.DataFrame()
    selected_ml_cfg = cfg
    if cfg.ml_enabled:
        if cfg.ml_grid_search_enabled:
            ml_grid_df = run_ml_parameter_grid_search(
                train_port_rets=train_port_rets,
                test_port_rets=test_port_rets,
                cfg=cfg,
                hmm_engine=hmm_engine,
            )
            valid_df = ml_grid_df[ml_grid_df.get("test_sharpe_ratio").notna()] if not ml_grid_df.empty else pd.DataFrame()
            if not valid_df.empty:
                best = valid_df.iloc[0]
                selected_ml_cfg = replace(
                    cfg,
                    rf_n_estimators=int(best["rf_n_estimators"]),
                    rf_max_depth=int(best["rf_max_depth"]),
                    rf_min_samples_leaf=int(best["rf_min_samples_leaf"]),
                    ml_target_exposure=float(best["ml_target_exposure"]),
                    hmm_n_states=int(best["hmm_n_states"]),
                    ml_auto_threshold=True,
                )
                print(
                    "Selected ML params from grid: "
                    f"rf_n_estimators={selected_ml_cfg.rf_n_estimators}, "
                    f"rf_max_depth={selected_ml_cfg.rf_max_depth}, "
                    f"rf_min_samples_leaf={selected_ml_cfg.rf_min_samples_leaf}, "
                    f"ml_target_exposure={selected_ml_cfg.ml_target_exposure}, "
                    f"hmm_n_states={selected_ml_cfg.hmm_n_states}"
                )

        ml_train_rets, ml_test_rets, ml_diag, ml_state_df = apply_hmm_softprob_rf_strategy(
            train_port_rets=train_port_rets,
            test_port_rets=test_port_rets,
            cfg=selected_ml_cfg,
            hmm_engine=hmm_engine,
        )
        _, ml_train_metrics = evaluate_weighted_portfolio(
            pd.DataFrame({"p": ml_train_rets}),
            pd.Series({"p": 1.0}),
        )
        _, ml_test_metrics = evaluate_weighted_portfolio(
            pd.DataFrame({"p": ml_test_rets}),
            pd.Series({"p": 1.0}),
        )

    # First, apply ML-specific position scalar (only affects ML-filtered returns)
    ml_train_rets, ml_test_rets, ml_pos_diag = compute_and_apply_ml_position_scalar(
        cfg=selected_ml_cfg,
        ml_train_rets=ml_train_rets,
        ml_test_rets=ml_test_rets,
        train_port_rets=train_port_rets,
        test_port_rets=test_port_rets,
        hmm_engine=hmm_engine,
    )

    # Persist ML scalar diagnostics
    pd.DataFrame([ml_pos_diag]).to_csv(here / "ml_position_scalar_diagnostics.csv", index=False)

    # Apply vol-target scaling (train-based) if enabled. This will scale both train/test and ML-filtered portfolios
    scaled_train, scaled_test, scaled_ml_train, scaled_ml_test, vol_diag = compute_and_apply_vol_target(
        cfg=cfg,
        train_port_rets=train_port_rets,
        test_port_rets=test_port_rets,
        ml_train_rets=ml_train_rets,
        ml_test_rets=ml_test_rets,
    )

    # Replace variables with scaled versions so subsequent metrics and CSVs reflect scaling
    train_port_rets = scaled_train
    test_port_rets = scaled_test
    ml_train_rets = scaled_ml_train
    ml_test_rets = scaled_ml_test

    # Recompute portfolio-level metrics after scaling
    _, train_metrics = evaluate_weighted_portfolio(pd.DataFrame({"p": train_port_rets}), pd.Series({"p": 1.0}))
    _, test_metrics = evaluate_weighted_portfolio(pd.DataFrame({"p": test_port_rets}), pd.Series({"p": 1.0}))
    _, ml_train_metrics = evaluate_weighted_portfolio(pd.DataFrame({"p": ml_train_rets}), pd.Series({"p": 1.0}))
    _, ml_test_metrics = evaluate_weighted_portfolio(pd.DataFrame({"p": ml_test_rets}), pd.Series({"p": 1.0}))

    # Persist diagnostics about the vol-target scaling
    pd.DataFrame([vol_diag]).to_csv(here / "vol_target_diagnostics.csv", index=False)

    out_summary = here / "asset_strategy_summary.csv"
    out_weights = here / "black_litterman_weights.csv"
    out_train_rets = here / "portfolio_train_returns.csv"
    out_test_rets = here / "portfolio_test_returns.csv"
    out_train_ml_rets = here / "portfolio_train_returns_ml.csv"
    out_test_ml_rets = here / "portfolio_test_returns_ml.csv"
    out_ml_states = here / "ml_hmm_state_params.csv"
    out_ml_diag = here / "ml_diagnostics.csv"
    out_ml_grid = here / "ml_parameter_grid_results.csv"
    out_ml_grid_top = here / "ml_parameter_grid_top10.csv"
    out_chart_train = here / "equity_curve_train_comparison.png"
    out_chart_test = here / "equity_curve_test_comparison.png"
    out_metrics = here / "bl_portfolio_metrics.csv"

    summary_df.to_csv(out_summary, index=False)
    bl_weights.rename("weight").to_csv(out_weights, header=True)
    train_port_rets.rename("portfolio_return").to_csv(out_train_rets, header=True)
    test_port_rets.rename("portfolio_return").to_csv(out_test_rets, header=True)
    ml_train_rets.rename("portfolio_return_ml").to_csv(out_train_ml_rets, header=True)
    ml_test_rets.rename("portfolio_return_ml").to_csv(out_test_ml_rets, header=True)
    if not ml_state_df.empty:
        ml_state_df.to_csv(out_ml_states, index=False)
    if ml_diag:
        pd.DataFrame([ml_diag]).to_csv(out_ml_diag, index=False)
    if not ml_grid_df.empty:
        ml_grid_df.to_csv(out_ml_grid, index=False)
        if "test_sharpe_ratio" in ml_grid_df.columns:
            ml_grid_df.head(10).to_csv(out_ml_grid_top, index=False)

    eq_strategy_train = train_returns_df.mean(axis=1)
    eq_strategy_test = test_returns_df.mean(axis=1)
    eq_hold_train = train_bh_returns_df.mean(axis=1)
    eq_hold_test = test_bh_returns_df.mean(axis=1)

    save_equity_curve_chart(
        {
            "BL Portfolio": train_port_rets,
            "ML Filtered Portfolio": ml_train_rets,
            "Equal-Weight Strategy Basket": eq_strategy_train,
            "Equal Hold Assets": eq_hold_train,
        },
        out_chart_train,
        "Train Equity Curves",
    )
    save_equity_curve_chart(
        {
            "BL Portfolio": test_port_rets,
            "ML Filtered Portfolio": ml_test_rets,
            "Equal-Weight Strategy Basket": eq_strategy_test,
            "Equal Hold Assets": eq_hold_test,
        },
        out_chart_test,
        "Test Equity Curves",
    )

    metrics_df = pd.DataFrame(
        [
            {"dataset": "train", **train_metrics},
            {"dataset": "test", **test_metrics},
            {"dataset": "train_ml", **ml_train_metrics},
            {"dataset": "test_ml", **ml_test_metrics},
        ]
    )
    metrics_df.to_csv(out_metrics, index=False)

    print("=" * 70)
    print("MULTI-ASSET PIPELINE DONE (NO PLOTS)")
    print("=" * 70)
    print(f"Valid assets: {len(summary_df)}")
    print("Top 10 BL weights:")
    for asset, w in bl_weights.head(10).items():
        print(f"  {asset:<20} {w:>8.2%}")

    print("Train metrics:")
    print(train_metrics)
    print("Test metrics:")
    print(test_metrics)
    if cfg.ml_enabled:
        print("Train metrics (ML filtered):")
        print(ml_train_metrics)
        print("Test metrics (ML filtered):")
        print(ml_test_metrics)
        if ml_diag:
            print("ML diagnostics:")
            print(ml_diag)

    print("Saved:")
    print(f"  {out_summary}")
    print(f"  {out_weights}")
    print(f"  {out_train_rets}")
    print(f"  {out_test_rets}")
    print(f"  {out_train_ml_rets}")
    print(f"  {out_test_ml_rets}")
    if cfg.ml_enabled:
        print(f"  {out_ml_states}")
        print(f"  {out_ml_diag}")
        if cfg.ml_grid_search_enabled:
            print(f"  {out_ml_grid}")
            print(f"  {out_ml_grid_top}")
    print(f"  {out_chart_train}")
    print(f"  {out_chart_test}")
    print(f"  {out_metrics}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
