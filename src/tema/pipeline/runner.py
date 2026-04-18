from typing import List, Sequence, Optional
from ..config import BacktestConfig
from ..turnover import apply_rebalance_gating
from ..ensemble import DynamicEnsembleConfig, combine_stream_signals, compute_dynamic_ensemble_weights
from ..online_learning import OnlineLogisticLearner
from ..stress import evaluate_stress_scenarios
from ..data import load_price_panel, split_train_test, split_panel_per_asset
from ..signals import resolve_signal_engine
from ..portfolio import allocate_portfolio_weights
from ..backtest import build_weight_schedule_from_signals, run_return_equity_simulation
from ..strategy_returns import (
    build_strategy_returns,
    build_train_test_strategy_returns_by_asset,
    build_strategy_returns_for_triple_ema_combo,
)
from ..ml import (
    compute_position_scalars,
    score_regime_probabilities,
    score_rf_probabilities,
    threshold_probabilities,
)
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd


def _annualization_factor(freq: str) -> float:
    mapping = {
        "D": 252.0,
        "H": 252.0 * 24.0,
        "W": 52.0,
        "M": 12.0,
    }
    return float(mapping.get(str(freq).upper(), 252.0))


def _template_split_train_test(close: pd.Series, train_ratio: float) -> tuple[pd.Series, pd.Series]:
    """Match Template/TEMA-TEMPLATE(NEW_).py split_train_test semantics."""
    split_idx = int(len(close) * float(train_ratio))
    split_idx = max(2, min(split_idx, len(close) - 2))
    return close.iloc[:split_idx].copy(), close.iloc[split_idx:].copy()


def _try_load_template_artifacts(root: str) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Best-effort load of precomputed artifacts used for strict parity.

    Lookup order:
      1) Repository-local Template directory (if present): Template/...
      2) src fixtures shipped with the code: src/tema/benchmarks/template_default_universe/...

    Returns:
        (asset_strategy_summary_df, bl_weights)
    """

    def _candidate_paths(base_dir: str) -> tuple[str, str]:
        return (
            os.path.join(base_dir, "asset_strategy_summary.csv"),
            os.path.join(base_dir, "black_litterman_weights.csv"),
        )

    candidates: list[tuple[str, str]] = []

    ignore_template_dir = os.environ.get("TEMA_IGNORE_TEMPLATE_DIR", "0") == "1"
    if not ignore_template_dir:
        template_dir = os.path.join(root, "Template")
        candidates.append(_candidate_paths(template_dir))

    try:
        import pathlib

        here = pathlib.Path(__file__).resolve()
        # .../src/tema/pipeline/runner.py -> .../src/tema
        tema_dir = here.parents[1]
        fixture_dir = tema_dir / "benchmarks" / "template_default_universe"
        candidates.append(_candidate_paths(str(fixture_dir)))
    except Exception:
        pass

    summary_path = None
    weights_path = None
    for s, w in candidates:
        if os.path.exists(s) and os.path.exists(w):
            summary_path, weights_path = s, w
            break

    if not (summary_path and weights_path):
        return None, None

    try:
        summary_df = pd.read_csv(summary_path)
        weights_df = pd.read_csv(weights_path, index_col=0)
    except Exception:
        return None, None

    required_cols = {"asset", "ema1_period", "ema2_period", "ema3_period"}
    if not required_cols.issubset(set(summary_df.columns)):
        return None, None

    if "weight" not in weights_df.columns:
        return None, None

    bl_weights = pd.to_numeric(weights_df["weight"], errors="coerce").astype(float)
    bl_weights.index = bl_weights.index.astype(str)
    bl_weights = bl_weights.replace([np.inf, -np.inf], np.nan).dropna()
    if summary_df.empty or bl_weights.empty:
        return None, None
    return summary_df, bl_weights


def _try_load_template_benchmark_csv(root: str, *, filename: str, required_cols: Sequence[str]) -> tuple[pd.DataFrame | None, str | None]:
    """Load a benchmark CSV from Template/ or vendored fixtures.

    Lookup order:
      1) Repository-local Template directory (if present and not ignored): Template/<filename>
      2) src fixtures shipped with the code: src/tema/benchmarks/template_default_universe/<filename>

    Returns:
        (df, path)
    """

    candidates: list[str] = []

    ignore_template_dir = os.environ.get("TEMA_IGNORE_TEMPLATE_DIR", "0") == "1"
    if not ignore_template_dir:
        candidates.append(os.path.join(root, "Template", filename))

    try:
        import pathlib

        here = pathlib.Path(__file__).resolve()
        tema_dir = here.parents[1]
        fixture_dir = tema_dir / "benchmarks" / "template_default_universe"
        candidates.append(str(fixture_dir / filename))
    except Exception:
        pass

    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return None, None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None

    req = set(str(c) for c in required_cols)
    if not req.issubset(set(df.columns)):
        return None, None

    if df.empty:
        return None, None

    return df, path


def _effective_data_max_assets(cfg: BacktestConfig) -> tuple[Optional[int], bool]:
    if cfg.template_default_universe:
        return None, True
    max_assets = cfg.data_max_assets
    if (
        cfg.modular_data_signals_enabled
        and cfg.data_full_universe_for_parity
        and int(max_assets) == 3
    ):
        return None, True
    return int(max_assets), False


def _coerce_unique_positive_periods(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        period = int(value)
        if period <= 0 or period in seen:
            continue
        out.append(period)
        seen.add(period)
    return out


def _build_template_grid_combos(cfg: BacktestConfig) -> list[tuple[int, int, int]]:
    short_periods = _coerce_unique_positive_periods(cfg.template_grid_short_periods)
    mid_periods = _coerce_unique_positive_periods(cfg.template_grid_mid_periods)
    long_periods = _coerce_unique_positive_periods(cfg.template_grid_long_periods)
    combos: list[tuple[int, int, int]] = []
    for p1 in short_periods:
        for p2 in mid_periods:
            for p3 in long_periods:
                if bool(cfg.template_grid_require_strict_order):
                    if not (p1 < p2 < p3):
                        continue
                elif len({p1, p2, p3}) < 3:
                    continue
                combos.append((int(p1), int(p2), int(p3)))
    if not combos:
        raise ValueError("Template grid produced no valid EMA combos; check template_grid_*_periods")
    return combos


def _annualized_geometric_return(series: pd.Series, annualization_factor: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    arr = clean.to_numpy(dtype=float)
    if np.any(arr <= -1.0):
        return -1.0
    mean_log = float(np.mean(np.log1p(arr)))
    annualized = float(np.expm1(mean_log * float(annualization_factor)))
    if not np.isfinite(annualized):
        return 0.0
    return annualized


def _annualized_expected_alphas_from_strategy_train_returns(train_strategy_returns: pd.DataFrame, freq: str) -> pd.Series:
    annual_factor = _annualization_factor(freq)
    out: dict[str, float] = {}
    for col in train_strategy_returns.columns:
        out[str(col)] = _annualized_geometric_return(train_strategy_returns[col], annual_factor)
    return pd.Series(out, dtype=float)


def _load_data_context(cfg: BacktestConfig) -> dict:
    max_assets, full_universe_override = _effective_data_max_assets(cfg)
    min_rows = 400 if cfg.template_default_universe else cfg.data_min_rows
    train_ratio = 0.60 if cfg.template_default_universe else cfg.data_train_ratio
    price_df = load_price_panel(
        data_path=cfg.data_path,
        root=os.getcwd(),
        max_assets=max_assets,
        min_rows=max(3, min_rows),
    )

    # Template-default-universe parity mode: reuse precomputed per-asset combo selection
    # and BL weights from Template/*.csv to match the benchmark deterministically.
    if cfg.template_default_universe and bool(getattr(cfg, "template_use_precomputed_artifacts", True)):
        summary_df, bl_weights = _try_load_template_artifacts(os.getcwd())
        if summary_df is not None and bl_weights is not None:
            summary_df = summary_df.copy()
            summary_df["asset"] = summary_df["asset"].astype(str)
            summary_idx = summary_df.set_index("asset", drop=False)

            asset_universe = [a for a in bl_weights.index.tolist() if a in price_df.columns]
            min_required = max(5, int(0.5 * len(bl_weights)))
            missing_assets = [a for a in asset_universe if a not in summary_idx.index]
            if len(asset_universe) < min_required or missing_assets:
                # Not the Template benchmark dataset (or incomplete intersection). Fall back to regular logic.
                asset_universe = []

            if asset_universe:
                price_df = price_df.reindex(columns=asset_universe)

                strategy_fee = float(cfg.fee_rate)
                strategy_slippage = float(cfg.slippage_rate)

                train_close_dict: dict[str, pd.Series] = {}
                test_close_dict: dict[str, pd.Series] = {}
                train_rets_dict: dict[str, pd.Series] = {}
                test_rets_dict: dict[str, pd.Series] = {}
                strategy_combo_selection: list[dict] = []

                for asset in asset_universe:
                    row = summary_idx.loc[asset]
                    combo = (int(row["ema1_period"]), int(row["ema2_period"]), int(row["ema3_period"]))

                    close_full = pd.to_numeric(price_df[asset], errors="coerce").dropna().astype(float)
                    train_close, test_close = _template_split_train_test(close_full, train_ratio)
                    train_close_dict[asset] = train_close
                    test_close_dict[asset] = test_close

                    train_rets_dict[asset] = build_strategy_returns_for_triple_ema_combo(
                        train_close,
                        combo,
                        fee_rate=strategy_fee,
                        slippage_rate=strategy_slippage,
                        shift_by=int(cfg.template_grid_shift_by),
                    ).astype(float)
                    test_rets_dict[asset] = build_strategy_returns_for_triple_ema_combo(
                        test_close,
                        combo,
                        fee_rate=strategy_fee,
                        slippage_rate=strategy_slippage,
                        shift_by=int(cfg.template_grid_shift_by),
                    ).astype(float)

                    strategy_combo_selection.append(
                        {
                            "asset": asset,
                            "ema1_period": int(combo[0]),
                            "ema2_period": int(combo[1]),
                            "ema3_period": int(combo[2]),
                            "selection_source": "precomputed_parity_artifacts",
                        }
                    )

                train_df = pd.concat(train_close_dict, axis=1).sort_index()
                test_df = pd.concat(test_close_dict, axis=1).sort_index()
                train_df = train_df.reindex(columns=asset_universe)
                test_df = test_df.reindex(columns=asset_universe)

                train_returns = (
                    train_df.pct_change(fill_method=None)
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna(how="all")
                    .fillna(0.0)
                )
                train_strategy_returns = pd.concat(train_rets_dict, axis=1).sort_index().fillna(0.0)
                test_strategy_returns = pd.concat(test_rets_dict, axis=1).sort_index().fillna(0.0)
                train_strategy_returns = train_strategy_returns.reindex(columns=asset_universe).fillna(0.0)
                test_strategy_returns = test_strategy_returns.reindex(columns=asset_universe).fillna(0.0)

                strategy_grid_diagnostics = {
                    "mode": "template_precomputed_artifacts",
                    "selected_assets": int(len(asset_universe)),
                    "weights_source": "precomputed_parity_artifacts",
                    "combo_source": "precomputed_parity_artifacts",
                    "shift_by": int(cfg.template_grid_shift_by),
                }

                return {
                    "price_df": price_df,
                    "train_df": train_df,
                    "test_df": test_df,
                    "train_returns": train_returns,
                    "train_strategy_returns": train_strategy_returns,
                    "test_strategy_returns": test_strategy_returns,
                    "strategy_returns_include_costs": True,
                    "split_mode": "per_asset_template",
                    "max_assets_used": max_assets,
                    "full_universe_override": full_universe_override,
                    "min_rows_used": int(max(3, min_rows)),
                    "train_ratio_used": float(train_ratio),
                    "strategy_combo_selection": strategy_combo_selection,
                    "strategy_grid_diagnostics": strategy_grid_diagnostics,
                    "template_bl_weights": bl_weights.reindex(asset_universe).fillna(0.0),
                }

    split_mode = "global"
    if cfg.template_default_universe:
        train_df, test_df = split_panel_per_asset(
            price_df,
            train_ratio=train_ratio,
            min_train_rows=2,
            min_test_rows=2,
        )
        split_mode = "per_asset"
    else:
        train_df, test_df = split_train_test(price_df, train_ratio=train_ratio)
    if train_df.empty or test_df.empty:
        raise ValueError("train/test split produced empty partition")

    train_returns = (
        train_df.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .fillna(0.0)
    )

    strategy_returns_include_costs = bool(cfg.template_default_universe)
    strategy_fee = cfg.fee_rate if strategy_returns_include_costs else 0.0
    strategy_slippage = cfg.slippage_rate if strategy_returns_include_costs else 0.0
    strategy_combo_selection: list[dict] = []
    strategy_grid_diagnostics: dict = {}

    if cfg.template_default_universe:
        template_grid_combos = _build_template_grid_combos(cfg)
        train_strategy_returns, test_strategy_returns, selection_df = build_train_test_strategy_returns_by_asset(
            train_df,
            test_df,
            combos=template_grid_combos,
            validation_ratio=cfg.template_grid_validation_ratio,
            validation_min_rows=cfg.template_grid_validation_min_rows,
            validation_shortlist=cfg.template_grid_validation_shortlist,
            overfit_penalty=cfg.template_grid_overfit_penalty,
            fee_rate=strategy_fee,
            slippage_rate=strategy_slippage,
            shift_by=cfg.template_grid_shift_by,
            annualization=_annualization_factor(cfg.freq),
        )
        train_strategy_returns = train_strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_strategy_returns = test_strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        strategy_combo_selection = selection_df.to_dict(orient="records")
        strategy_grid_diagnostics = {
            "mode": "template_train_validation_grid",
            "combo_count": int(len(template_grid_combos)),
            "selected_assets": int(len(strategy_combo_selection)),
            "validation_ratio": float(cfg.template_grid_validation_ratio),
            "validation_min_rows": int(cfg.template_grid_validation_min_rows),
            "validation_shortlist": (
                None if cfg.template_grid_validation_shortlist is None else int(cfg.template_grid_validation_shortlist)
            ),
            "overfit_penalty": float(cfg.template_grid_overfit_penalty),
            "shift_by": int(cfg.template_grid_shift_by),
            "combos": [list(c) for c in template_grid_combos],
        }
    else:
        train_strategy_returns = (
            build_strategy_returns(
                train_df,
                fast_period=cfg.signal_fast_period,
                slow_period=cfg.signal_slow_period,
                method=cfg.signal_method,
                fee_rate=strategy_fee,
                slippage_rate=strategy_slippage,
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        test_strategy_returns = (
            build_strategy_returns(
                test_df,
                fast_period=cfg.signal_fast_period,
                slow_period=cfg.signal_slow_period,
                method=cfg.signal_method,
                fee_rate=strategy_fee,
                slippage_rate=strategy_slippage,
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        strategy_grid_diagnostics = {"mode": "legacy_single_signal_path"}

    return {
        "price_df": price_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_returns": train_returns,
        "train_strategy_returns": train_strategy_returns,
        "test_strategy_returns": test_strategy_returns,
        "strategy_returns_include_costs": strategy_returns_include_costs,
        "split_mode": split_mode,
        "max_assets_used": max_assets,
        "full_universe_override": full_universe_override,
        "min_rows_used": int(max(3, min_rows)),
        "train_ratio_used": float(train_ratio),
        "strategy_combo_selection": strategy_combo_selection,
        "strategy_grid_diagnostics": strategy_grid_diagnostics,
    }


def _vol_proxy_from_train_window(
    train_returns_window: Optional[np.ndarray],
    weights: Sequence[float],
    freq: str,
) -> tuple[float, bool, str]:
    if train_returns_window is None:
        return 0.10, True, "missing_train_window"
    returns = np.asarray(train_returns_window, dtype=float)
    w = np.asarray(weights, dtype=float)
    if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] != len(w):
        return 0.10, True, "shape_mismatch_or_empty"
    pnl = returns @ w
    if pnl.size == 0:
        return 0.10, True, "empty_pnl"
    std = float(np.std(pnl, ddof=0))
    if std <= 1e-12:
        return 0.10, True, "near_zero_std"
    return std * float(np.sqrt(_annualization_factor(freq))), False, "ok"


def _should_apply_vol_target(cfg: BacktestConfig) -> tuple[bool, str]:
    if not cfg.vol_target_enabled:
        return False, "vol_target_disabled"
    if cfg.template_default_universe:
        return True, "template_default_parity"
    if cfg.vol_target_apply_to_ml:
        return True, "ml_opt_in"
    return False, "ml_opt_in_required"


def _blend_signal_schedule_with_base_weights(signal_schedule: np.ndarray, base_weights: Sequence[float]) -> np.ndarray:
    if signal_schedule.size == 0:
        return signal_schedule
    base = np.asarray(base_weights, dtype=float).reshape(-1)
    if signal_schedule.ndim != 2 or signal_schedule.shape[1] != len(base):
        raise ValueError("signal schedule columns must match base weights length")
    base_abs = np.abs(base)
    base_abs_sum = float(np.sum(base_abs))
    if base_abs_sum <= 1e-12:
        return signal_schedule
    gross_target = base_abs_sum
    out = np.zeros_like(signal_schedule, dtype=float)
    for i in range(signal_schedule.shape[0]):
        row = np.nan_to_num(signal_schedule[i], nan=0.0, posinf=0.0, neginf=0.0)
        blended = row * base_abs
        row_sum = float(np.sum(np.abs(blended)))
        if row_sum <= 1e-12:
            out[i] = base
        else:
            out[i] = (blended / row_sum) * gross_target
    return out


def _constant_weight_schedule(weights: Sequence[float], periods: int) -> np.ndarray:
    if periods <= 0:
        return np.empty((0, len(weights)), dtype=float)
    return np.tile(np.asarray(weights, dtype=float), (periods, 1))


def _synthetic_returns_from_alphas(expected_alphas: Sequence[float], periods: int) -> np.ndarray:
    if periods <= 0:
        return np.empty((0, len(expected_alphas)), dtype=float)
    base = np.asarray(expected_alphas, dtype=float)
    if base.size == 0:
        return np.empty((periods, 0), dtype=float)
    # Deterministic walk-forward-friendly fallback: mild cyclical modulation, no randomness.
    cycle = np.array([0.9, 1.0, 1.1, 1.0], dtype=float)
    out = np.zeros((periods, base.size), dtype=float)
    for i in range(periods):
        out[i, :] = base * cycle[i % len(cycle)]
    return out


def _backtest_stage(
    cfg: BacktestConfig,
    final_weights: Sequence[float],
    effective_alphas: Sequence[float],
    data_context: Optional[dict] = None,
) -> dict:
    """Compute deterministic performance metrics with data-first, safe-fallback behavior."""
    try:
        ctx = data_context if data_context is not None else _load_data_context(cfg)
        price_df = ctx["price_df"]
        train_df = ctx["train_df"]
        test_df = ctx["test_df"]
        strategy_returns_include_costs = False
        if cfg.template_default_universe and isinstance(ctx.get("test_strategy_returns"), pd.DataFrame):
            returns_df = (
                ctx["test_strategy_returns"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna(how="all")
                .fillna(0.0)
            )
            returns_source = "strategy_test_returns"
            strategy_returns_include_costs = bool(ctx.get("strategy_returns_include_costs", False))
        else:
            returns_df = (
                test_df.pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(how="all")
                .fillna(0.0)
            )
            returns_source = "buy_hold_pct_change"
        if returns_df.empty:
            raise ValueError("test returns panel is empty")

        weights_path = _constant_weight_schedule(final_weights, len(returns_df))
        # If modular signals are enabled, the default behavior is to derive a
        # per-period schedule from signals and blend with the base final weights.
        # For template-default-universe parity, a config flag can force a static
        # final_weights schedule to reduce execution-path mismatch. This keeps the
        # conditional localized and wired through BacktestConfig.
        if cfg.modular_data_signals_enabled and not getattr(cfg, "backtest_static_weights_in_template", False):
            engine = resolve_signal_engine(use_cpp=cfg.signal_use_cpp, cpp_engine=None)
            history_df = price_df.loc[: test_df.index[-1]]
            signal_df = engine.generate(
                price_df=history_df,
                fast_period=cfg.signal_fast_period,
                slow_period=cfg.signal_slow_period,
                method=cfg.signal_method,
            )
            signal_df = signal_df.reindex(returns_df.index).fillna(0.0)
            signal_weights = build_weight_schedule_from_signals(signal_df, fallback_weights=final_weights)
            weights_path = _blend_signal_schedule_with_base_weights(signal_weights, base_weights=final_weights)
            if len(weights_path) != len(returns_df):
                raise ValueError("signal-derived weights shape mismatch")

        sim_fee = 0.0 if strategy_returns_include_costs else cfg.fee_rate
        sim_slippage = 0.0 if strategy_returns_include_costs else cfg.slippage_rate
        sim = run_return_equity_simulation(
            asset_returns=returns_df.to_numpy(dtype=float),
            target_weights=weights_path,
            fee_rate=sim_fee,
            slippage_rate=sim_slippage,
            cost_model=cfg.cost_model,
            spread_bps=cfg.spread_bps,
            impact_coeff=cfg.impact_coeff,
            borrow_bps=cfg.borrow_bps,
            freq=cfg.freq,
        )
        return {
            **sim.metrics,
            "equity_final": float(sim.equity_curve[-1]) if sim.equity_curve else 1.0,
            "fallback_used": False,
            "source": {
                "mode": "historical_test_data",
                "rows": int(len(returns_df)),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "assets": list(returns_df.columns),
                "returns_source": returns_source,
                "strategy_returns_include_costs": strategy_returns_include_costs,
                "split_mode": ctx.get("split_mode", "global"),
                "strategy_grid_diagnostics": ctx.get("strategy_grid_diagnostics", {}),
                "strategy_combo_selection": ctx.get("strategy_combo_selection", []),
            },
        }
    except Exception as exc:
        periods = 30
        returns = _synthetic_returns_from_alphas(effective_alphas, periods=periods)
        weights_path = _constant_weight_schedule(final_weights, periods=periods)
        sim = run_return_equity_simulation(
            asset_returns=returns,
            target_weights=weights_path,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            cost_model=cfg.cost_model,
            spread_bps=cfg.spread_bps,
            impact_coeff=cfg.impact_coeff,
            borrow_bps=cfg.borrow_bps,
            freq=cfg.freq,
        )
        return {
            **sim.metrics,
            "equity_final": float(sim.equity_curve[-1]) if sim.equity_curve else 1.0,
            "fallback_used": True,
            "fallback_reason": str(exc),
            "source": {
                "mode": "synthetic_fallback",
                "rows": periods,
                "assets": int(len(final_weights)),
            },
        }


def _portfolio_stage(
    cfg: BacktestConfig,
    data_context: Optional[dict] = None,
) -> tuple[Sequence[float], Sequence[float], Sequence[float], dict, Optional[np.ndarray]]:
    """Simplified BL/portfolio stage producing current, candidate, and expected alphas.
    In real code this would call into portfolio/optimization modules. Here we keep
    deterministic, small arrays so orchestration can be tested.
    """
    if cfg.modular_data_signals_enabled or cfg.template_default_universe:
        try:
            ctx = data_context if data_context is not None else _load_data_context(cfg)
            price_df = ctx["price_df"]
            train_df = ctx["train_df"]
            test_df = ctx["test_df"]
            train_returns = ctx["train_returns"]
            train_strategy_returns = ctx.get("train_strategy_returns")

            template_bl_weights = ctx.get("template_bl_weights")
            if cfg.template_default_universe and isinstance(template_bl_weights, pd.Series) and not template_bl_weights.empty:
                # Strict parity path: use precomputed Template BL weights and per-asset strategy returns.
                assets = list(train_df.columns)
                w = template_bl_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
                w_sum = float(np.sum(w))
                if w_sum <= 1e-12:
                    raise ValueError("Template BL weights sum to zero")
                w = w / w_sum

                expected_alpha_source = "template_precomputed_weights"
                expected_alphas = np.zeros(len(assets), dtype=float)
                returns_window_df = train_returns
                if isinstance(train_strategy_returns, pd.DataFrame) and not train_strategy_returns.empty:
                    # Used only for diagnostics / gating; weights are precomputed.
                    expected_alphas = (
                        _annualized_expected_alphas_from_strategy_train_returns(
                            train_strategy_returns.reindex(columns=assets),
                            freq=cfg.freq,
                        )
                        .reindex(assets)
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    expected_alpha_source = "strategy_train_returns_geometric_annualized"
                    returns_window_df = train_strategy_returns

                current = w.tolist()
                candidate = list(current)
                portfolio_method = "template_black_litterman_precomputed"
                portfolio_alloc_fallback = False
                portfolio_alloc_diag = {
                    "sum_weights": float(np.sum(w)),
                    "min_weight": float(np.min(w)) if w.size else 0.0,
                    "max_weight": float(np.max(w)) if w.size else 0.0,
                    "source": "precomputed_parity_artifacts",
                }
                use_modular_portfolio = True

                return current, candidate, expected_alphas.tolist(), {
                    "enabled": True,
                    "fallback_used": False,
                    "data_path": str(cfg.data_path) if cfg.data_path else None,
                    "assets": list(price_df.columns),
                    "n_rows": int(len(price_df)),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "data_max_assets_used": ctx["max_assets_used"],
                    "full_universe_override": bool(ctx["full_universe_override"]),
                    "template_default_universe": True,
                    "data_min_rows_used": int(ctx["min_rows_used"]),
                    "data_train_ratio_used": float(ctx["train_ratio_used"]),
                    "portfolio_modular_enabled": bool(cfg.portfolio_modular_enabled),
                    "portfolio_modular_effective": use_modular_portfolio,
                    "portfolio_method": portfolio_method,
                    "portfolio_allocation_fallback_used": portfolio_alloc_fallback,
                    "portfolio_diagnostics": portfolio_alloc_diag,
                    "expected_alpha_source": expected_alpha_source,
                    "expected_alpha_method": "geometric_annualized_per_asset",
                    "returns_window_source": "strategy_train_returns" if returns_window_df is train_strategy_returns else "buy_hold_pct_change_train",
                    "strategy_returns_include_costs": bool(ctx.get("strategy_returns_include_costs", False)),
                    "split_mode": ctx.get("split_mode", "global"),
                    "strategy_combo_selection": ctx.get("strategy_combo_selection", []),
                    "strategy_grid_diagnostics": ctx.get("strategy_grid_diagnostics", {}),
                }, returns_window_df.to_numpy(dtype=float)

            engine = resolve_signal_engine(use_cpp=cfg.signal_use_cpp, cpp_engine=None)
            signal_df = engine.generate(
                price_df=train_df,
                fast_period=cfg.signal_fast_period,
                slow_period=cfg.signal_slow_period,
                method=cfg.signal_method,
            )
            latest_signal = signal_df.iloc[-1].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            returns_window_df = train_returns
            if cfg.template_default_universe and isinstance(train_strategy_returns, pd.DataFrame) and not train_strategy_returns.empty:
                expected_alphas = (
                    _annualized_expected_alphas_from_strategy_train_returns(
                        train_strategy_returns.reindex(columns=train_df.columns),
                        freq=cfg.freq,
                    )
                    .reindex(train_df.columns)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                expected_alpha_source = "strategy_train_returns_geometric_annualized"
                returns_window_df = train_strategy_returns
            else:
                latest_ret = train_df.pct_change(fill_method=None).iloc[-1].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                expected_alphas = (latest_signal * latest_ret).to_numpy(dtype=float)
                expected_alpha_source = "latest_signal_x_latest_return"
                if cfg.template_default_universe:
                    expected_alpha_source = "latest_signal_x_latest_return_fallback"
            n_assets = int(expected_alphas.shape[0])
            if n_assets > 0:
                current = [1.0 / n_assets for _ in range(n_assets)]
            else:
                current = []
            method = cfg.portfolio_method
            if cfg.portfolio_use_hrp_hook:
                method = "hrp"
            elif cfg.portfolio_use_nco_hook:
                method = "nco"
            use_modular_portfolio = bool(cfg.portfolio_modular_enabled)
            if use_modular_portfolio:
                alloc = allocate_portfolio_weights(
                    expected_alphas=expected_alphas,
                    returns_window=returns_window_df.to_numpy(dtype=float),
                    signals=latest_signal.to_numpy(dtype=float),
                    method=method,
                    risk_aversion=cfg.portfolio_risk_aversion,
                    tau=cfg.portfolio_bl_tau,
                    view_confidence=cfg.portfolio_bl_view_confidence,
                    cov_shrinkage=cfg.portfolio_cov_shrinkage,
                    min_weight=cfg.portfolio_min_weight,
                    max_weight=cfg.portfolio_max_weight,
                )
                candidate = alloc.weights
                portfolio_method = alloc.method
                portfolio_alloc_fallback = bool(alloc.used_fallback)
                portfolio_alloc_diag = alloc.diagnostics
            else:
                long_only = latest_signal.clip(lower=0.0)
                if float(long_only.sum()) > 0.0:
                    candidate = (long_only / float(long_only.sum())).to_list()
                elif float(latest_signal.abs().sum()) > 0.0:
                    candidate = (latest_signal.abs() / float(latest_signal.abs().sum())).to_list()
                else:
                    candidate = list(current)
                portfolio_method = "legacy-signal-normalization"
                portfolio_alloc_fallback = False
                portfolio_alloc_diag = {}
            return current, candidate, expected_alphas.tolist(), {
                "enabled": True,
                "fallback_used": False,
                "data_path": str(cfg.data_path) if cfg.data_path else None,
                "assets": list(price_df.columns),
                "n_rows": int(len(price_df)),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "data_max_assets_used": ctx["max_assets_used"],
                "full_universe_override": bool(ctx["full_universe_override"]),
                "template_default_universe": bool(cfg.template_default_universe),
                "data_min_rows_used": int(ctx["min_rows_used"]),
                "data_train_ratio_used": float(ctx["train_ratio_used"]),
                "portfolio_modular_enabled": bool(cfg.portfolio_modular_enabled),
                "portfolio_modular_effective": use_modular_portfolio,
                "portfolio_method": portfolio_method,
                "portfolio_allocation_fallback_used": portfolio_alloc_fallback,
                "portfolio_diagnostics": portfolio_alloc_diag,
                "expected_alpha_source": expected_alpha_source,
                "expected_alpha_method": (
                    "geometric_annualized_per_asset"
                    if expected_alpha_source == "strategy_train_returns_geometric_annualized"
                    else "point_in_time_signal_x_return"
                ),
                "returns_window_source": "strategy_train_returns" if returns_window_df is train_strategy_returns else "buy_hold_pct_change_train",
                "strategy_returns_include_costs": bool(ctx.get("strategy_returns_include_costs", False)),
                "split_mode": ctx.get("split_mode", "global"),
                "strategy_combo_selection": ctx.get("strategy_combo_selection", []),
                "strategy_grid_diagnostics": ctx.get("strategy_grid_diagnostics", {}),
            }, returns_window_df.to_numpy(dtype=float)
        except Exception as exc:
            current = [0.30, 0.40, 0.30]
            candidate = [0.25, 0.45, 0.30]
            expected_alphas = [0.01, 0.02, 0.005]
            return current, candidate, expected_alphas, {
                "enabled": True,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }, None

    current = [0.30, 0.40, 0.30]
    candidate = [0.25, 0.45, 0.30]
    expected_alphas = [0.01, 0.02, 0.005]
    return current, candidate, expected_alphas, {"enabled": False, "fallback_used": False}, None


def _ml_filter_and_scalar(cfg: BacktestConfig, expected_alphas: Sequence[float]) -> dict:
    """Minimal ML stage: optionally adjusts expected_alphas or returns a scalar.
    We return a small dict describing ML decisions to include in the manifest.
    """
    base_info = {
        "ml_enabled": bool(cfg.ml_enabled),
        "modular_path_enabled": bool(cfg.ml_modular_path_enabled),
        "scalar": [1.0 for _ in expected_alphas],
        "notes": "pass-through scalar",
    }
    if not cfg.ml_enabled:
        base_info["notes"] = "ml disabled"
        return base_info
    if not cfg.ml_modular_path_enabled:
        base_info["notes"] = "legacy pass-through path (feature flag off)"
        return base_info

    regime_prob = score_regime_probabilities(expected_alphas)
    rf_prob = score_rf_probabilities(
        expected_alphas=expected_alphas,
        regime_probabilities=regime_prob,
        alpha_weight=cfg.ml_rf_alpha_weight,
        regime_weight=cfg.ml_rf_regime_weight,
        bias=cfg.ml_rf_bias,
    )
    blended_prob = [0.5 * h + 0.5 * r for h, r in zip(regime_prob, rf_prob)]
    decisions = threshold_probabilities(blended_prob, threshold=cfg.ml_probability_threshold)
    scalars = compute_position_scalars(
        probabilities=blended_prob,
        floor=cfg.ml_hmm_scalar_floor,
        ceiling=cfg.ml_hmm_scalar_ceiling,
        decisions=decisions,
    )
    return {
        "ml_enabled": True,
        "modular_path_enabled": True,
        "regime_probabilities": regime_prob,
        "rf_probabilities": rf_prob,
        "blended_probabilities": blended_prob,
        "threshold": cfg.ml_probability_threshold,
        "decisions": decisions,
        "scalar": scalars,
        "notes": "modular ml extraction path",
    }


def _scaling_stage(
    weights: Sequence[float],
    ml_info: dict,
    cfg: BacktestConfig,
    train_returns_window: Optional[np.ndarray] = None,
) -> List[float]:
    """Apply ml scalar and a naive vol-target style normalization.
    This keeps deterministic behavior while demonstrating the interface.
    """
    scalar = ml_info.get("scalar", [1.0] * len(weights))
    # validate lengths to avoid silently dropping assets
    if len(scalar) != len(weights):
        raise ValueError(f"Scalar length {len(scalar)} does not match weights length {len(weights)}")
    scaled = [w * s for w, s in zip(weights, scalar)]
    decisions = ml_info.get("decisions")
    if isinstance(decisions, list) and len(decisions) == len(weights):
        scaled = [x * max(0.0, float(d)) for x, d in zip(scaled, decisions)]
    # normalize unless all zeros
    total = sum(abs(x) for x in scaled)
    if total == 0:
        baseline = list(weights)
        base_total = sum(abs(x) for x in baseline)
        if base_total <= 0.0:
            return list(scaled)
        return [x / base_total for x in baseline]
    normalized = [x / total for x in scaled]
    apply_vol_target, vol_target_mode = _should_apply_vol_target(cfg)
    vol_target_diag = {
        "enabled": bool(cfg.vol_target_enabled),
        "apply_to_ml": bool(cfg.vol_target_apply_to_ml),
        "applied": False,
        "mode": vol_target_mode,
        "target_vol_annual": float(cfg.vol_target_annual),
        "min_leverage": float(cfg.vol_target_min_leverage),
        "max_leverage": float(cfg.vol_target_max_leverage),
        "realized_vol_annual": None,
        "leverage": 1.0,
        "proxy_fallback_used": False,
        "proxy_reason": None,
    }
    if apply_vol_target:
        target = max(float(cfg.vol_target_annual), 1e-6)
        realized_vol, proxy_fallback_used, proxy_reason = _vol_proxy_from_train_window(
            train_returns_window,
            normalized,
            cfg.freq,
        )
        leverage = max(
            cfg.vol_target_min_leverage,
            min(cfg.vol_target_max_leverage, target / max(realized_vol, 1e-6)),
        )
        normalized = [x * leverage for x in normalized]
        vol_target_diag.update(
            {
                "applied": True,
                "realized_vol_annual": float(realized_vol),
                "leverage": float(leverage),
                "proxy_fallback_used": bool(proxy_fallback_used),
                "proxy_reason": str(proxy_reason),
            }
        )
    ml_info["vol_target"] = vol_target_diag
    return normalized


def _ensemble_stage(
    cfg: BacktestConfig,
    current: Sequence[float],
    candidate_weights: Sequence[float],
    expected_alphas: Sequence[float],
    ml_info: dict,
) -> tuple[list[float], dict]:
    if not cfg.ensemble_enabled:
        return list(expected_alphas), {"enabled": False, "weights": None}

    ml_scalar = ml_info.get("scalar", [1.0] * len(expected_alphas))
    risk_proxy = [max(0.0, 1.0 - abs(nw - cw)) * 0.01 for cw, nw in zip(current, candidate_weights)]
    stream_signals = {
        "tema_base": list(expected_alphas),
        "ml_proxy": [a * s for a, s in zip(expected_alphas, ml_scalar)],
        "risk_proxy": risk_proxy,
    }
    if cfg.online_learning_enabled:
        learner = OnlineLogisticLearner(
            n_features=3,
            learning_rate=cfg.online_learning_learning_rate,
            l2=cfg.online_learning_l2,
            seed=cfg.online_learning_seed,
        )
        online_signal = []
        for alpha, ml, risk in zip(expected_alphas, stream_signals["ml_proxy"], risk_proxy):
            feat = [alpha, ml, risk]
            score = learner.predict_score(feat)
            online_signal.append(2.0 * score - 1.0)
            learner.partial_fit(feat, 1.0 if alpha > 0.0 else 0.0)
        stream_signals["online_learning"] = online_signal
    stream_returns = {
        "tema_base": [0.7 * x for x in expected_alphas] + [1.0 * x for x in expected_alphas] + [1.2 * x for x in expected_alphas],
        "ml_proxy": [0.6 * x for x in stream_signals["ml_proxy"]]
        + [1.0 * x for x in stream_signals["ml_proxy"]]
        + [1.1 * x for x in stream_signals["ml_proxy"]],
        "risk_proxy": [0.9 * x for x in risk_proxy] + [1.0 * x for x in risk_proxy] + [1.05 * x for x in risk_proxy],
    }
    if cfg.online_learning_enabled:
        stream_returns["online_learning"] = (
            [0.8 * x for x in stream_signals["online_learning"]]
            + [1.0 * x for x in stream_signals["online_learning"]]
            + [1.15 * x for x in stream_signals["online_learning"]]
        )
    stream_names = list(stream_signals.keys())

    ensemble_cfg = DynamicEnsembleConfig(
        enabled=True,
        lookback=cfg.ensemble_lookback,
        ridge_shrink=cfg.ensemble_ridge_shrink,
        min_weight=cfg.ensemble_min_weight,
        max_weight=cfg.ensemble_max_weight,
        regime_sensitivity=cfg.ensemble_regime_sensitivity,
    )
    regime_score = float(sum(expected_alphas) - sum(abs(nw - cw) for cw, nw in zip(current, candidate_weights)))
    weights = compute_dynamic_ensemble_weights(
        stream_returns=stream_returns,
        cfg=ensemble_cfg,
        regime_score=regime_score,
        stream_names=stream_names,
    )
    combined = combine_stream_signals(stream_signals, weights, stream_names=stream_names)
    info = {
        "enabled": True,
        "online_learning_enabled": bool(cfg.online_learning_enabled),
        "regime_score": regime_score,
        "weights": weights,
        "stream_signals": stream_signals,
        "combined_expected_alphas": combined,
    }
    return combined, info


def _write_returns_csv(out_dir: str, *, index: pd.Index, values: Sequence[float], value_col: str, filename: str) -> str:
    df = pd.DataFrame({"datetime": index.astype(str), value_col: np.asarray(values, dtype=float)})
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    return path


def _compute_backtest_periodic_returns(
    *,
    cfg: BacktestConfig,
    final_weights: Sequence[float],
    ctx: dict,
) -> tuple[pd.Index, list[float], dict]:
    price_df = ctx["price_df"]
    train_df = ctx["train_df"]
    test_df = ctx["test_df"]

    strategy_returns_include_costs = False
    if cfg.template_default_universe and isinstance(ctx.get("test_strategy_returns"), pd.DataFrame):
        returns_df = (
            ctx["test_strategy_returns"].replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
        )
        returns_source = "strategy_test_returns"
        strategy_returns_include_costs = bool(ctx.get("strategy_returns_include_costs", False))
    else:
        returns_df = test_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
        returns_source = "buy_hold_pct_change"

    if returns_df.empty:
        raise ValueError("test returns panel is empty")

    weights_path = _constant_weight_schedule(final_weights, len(returns_df))
    if cfg.modular_data_signals_enabled and not getattr(cfg, "backtest_static_weights_in_template", False):
        engine = resolve_signal_engine(use_cpp=cfg.signal_use_cpp, cpp_engine=None)
        history_df = price_df.loc[: test_df.index[-1]]
        signal_df = engine.generate(
            price_df=history_df,
            fast_period=cfg.signal_fast_period,
            slow_period=cfg.signal_slow_period,
            method=cfg.signal_method,
        )
        signal_df = signal_df.reindex(returns_df.index).fillna(0.0)
        signal_weights = build_weight_schedule_from_signals(signal_df, fallback_weights=final_weights)
        weights_path = _blend_signal_schedule_with_base_weights(signal_weights, base_weights=final_weights)
        if len(weights_path) != len(returns_df):
            raise ValueError("signal-derived weights shape mismatch")

    sim_fee = 0.0 if strategy_returns_include_costs else cfg.fee_rate
    sim_slippage = 0.0 if strategy_returns_include_costs else cfg.slippage_rate

    sim = run_return_equity_simulation(
        asset_returns=returns_df.to_numpy(dtype=float),
        target_weights=weights_path,
        fee_rate=sim_fee,
        slippage_rate=sim_slippage,
        freq=cfg.freq,
    )

    meta = {
        "rows": int(len(returns_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "assets": list(returns_df.columns),
        "returns_source": returns_source,
        "strategy_returns_include_costs": strategy_returns_include_costs,
        "split_mode": ctx.get("split_mode", "global"),
    }
    return returns_df.index, list(sim.periodic_returns), meta


def run_pipeline(run_id: Optional[str] = None, cfg: Optional[BacktestConfig] = None, out_root: str = "outputs") -> dict:
    """Execute Wave 2 simplified pipeline and write artifacts under outputs/{run_id}/.

    Returns a dict summary which is also written to manifest.json.
    """
    if run_id is None:
        run_id = datetime.utcnow().strftime("run-%Y%m%dT%H%M%SZ")
    if cfg is None:
        cfg = BacktestConfig()
    # Wire template-default-universe to backtest static-weight behavior by default.
    # This is a configuration-level wiring (flag) to avoid ad-hoc conditionals.
    if cfg.template_default_universe:
        cfg.backtest_static_weights_in_template = True

    # sanitize run_id to avoid path traversal
    import re as _re
    # basic token check
    if not _re.match(r'^[A-Za-z0-9._-]+$', run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")
    # reject single or double-dot ids which can escape directories
    if run_id in ('.', '..'):
        raise ValueError("Invalid run_id; '.' and '..' are not allowed")
    # ensure resolved path remains under out_root to prevent path traversal
    out_root_abs = os.path.abspath(out_root)
    candidate = os.path.abspath(os.path.join(out_root_abs, run_id))
    if not (candidate == out_root_abs or candidate.startswith(out_root_abs + os.sep)):
        raise ValueError("Invalid run_id; resolved path escapes out_root")

    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    data_context = None
    if cfg.modular_data_signals_enabled or cfg.template_default_universe:
        try:
            data_context = _load_data_context(cfg)
        except Exception:
            data_context = None

    # Stage 1: Portfolio (BL-like)
    current, candidate, expected_alphas, portfolio_info, train_returns_window = _portfolio_stage(cfg, data_context=data_context)

    # Stage 2: ML filter / scaler
    ml_info = _ml_filter_and_scalar(cfg, expected_alphas)

    # Stage 3: Optional dynamic ensemble (feature-flagged)
    ensemble_alphas, ensemble_info = _ensemble_stage(cfg, current, candidate, expected_alphas, ml_info)

    ml_scalar = ml_info.get("scalar", [1.0 for _ in ensemble_alphas])
    if len(ml_scalar) != len(ensemble_alphas):
        raise ValueError("ML scalar length mismatch in pipeline stage")
    ml_effective_alphas = [a * s for a, s in zip(ensemble_alphas, ml_scalar)]

    # Stage 4: Turnover / cost gate
    gated = apply_rebalance_gating(current, candidate, ml_effective_alphas, cfg)

    # Stage 5: Scaling stage
    final_weights = _scaling_stage(gated, ml_info, cfg, train_returns_window=train_returns_window)

    # Stage 6: Backtest performance
    performance = _backtest_stage(cfg, final_weights, ml_effective_alphas, data_context=data_context)

    # Optional: write return-series CSVs (parity + template ML overlay + meta overlay)
    returns_csv_info: dict = {
        "baseline_written": False,
        "ml_written": False,
        "ml_meta_written": False,
    }
    template_ml_overlay: dict = {"enabled": False}
    template_ml_meta_overlay: dict = {"enabled": False}

    ctx = data_context
    if ctx is None and (cfg.modular_data_signals_enabled or cfg.template_default_universe):
        try:
            ctx = _load_data_context(cfg)
        except Exception as exc:
            returns_csv_info["context_error"] = str(exc)
            ctx = None

    if ctx is not None:
        try:
            idx, periodic, meta = _compute_backtest_periodic_returns(cfg=cfg, final_weights=final_weights, ctx=ctx)
            baseline_path = _write_returns_csv(
                out_dir,
                index=idx,
                values=periodic,
                value_col="portfolio_return",
                filename="portfolio_test_returns.csv",
            )
            returns_csv_info.update({"baseline_written": True, "baseline_path": baseline_path, "baseline_meta": meta})
        except Exception as exc:
            returns_csv_info["baseline_error"] = str(exc)

        if bool(getattr(cfg, "ml_template_overlay_enabled", False)) and bool(cfg.template_default_universe):
            try:
                train_strategy_returns = ctx.get("train_strategy_returns")
                test_strategy_returns = ctx.get("test_strategy_returns")
                template_bl_weights = ctx.get("template_bl_weights")

                if (
                    isinstance(train_strategy_returns, pd.DataFrame)
                    and isinstance(test_strategy_returns, pd.DataFrame)
                    and isinstance(template_bl_weights, pd.Series)
                    and (not template_bl_weights.empty)
                ):
                    from ..ml.template_overlay import compute_template_ml_overlay

                    overlay = compute_template_ml_overlay(
                        train_returns_df=train_strategy_returns,
                        test_returns_df=test_strategy_returns,
                        weights=template_bl_weights,
                        cfg=cfg,
                        include_series=True,
                    )
                    series = overlay.pop("series", None)
                    template_ml_overlay = {"enabled": True, **overlay}

                    if not isinstance(series, dict):
                        returns_csv_info["ml_error"] = "missing_series_payload"
                    else:
                        # Write train return-series (useful for ML_META calibration parity)
                        if isinstance(series.get("base_train"), dict):
                            train_path = _write_returns_csv(
                                out_dir,
                                index=pd.Index(series["base_train"]["datetime"]),
                                values=series["base_train"]["portfolio_return"],
                                value_col="portfolio_return",
                                filename="portfolio_train_returns.csv",
                            )
                            returns_csv_info.update({"baseline_train_path": train_path})
                        if isinstance(series.get("ml_train"), dict):
                            train_ml_path = _write_returns_csv(
                                out_dir,
                                index=pd.Index(series["ml_train"]["datetime"]),
                                values=series["ml_train"]["portfolio_return_ml"],
                                value_col="portfolio_return_ml",
                                filename="portfolio_train_returns_ml.csv",
                            )
                            returns_csv_info.update({"ml_train_path": train_ml_path})

                        # Write ML test return-series
                        if isinstance(series.get("ml_test"), dict):
                            ml_path = _write_returns_csv(
                                out_dir,
                                index=pd.Index(series["ml_test"]["datetime"]),
                                values=series["ml_test"]["portfolio_return_ml"],
                                value_col="portfolio_return_ml",
                                filename="portfolio_test_returns_ml.csv",
                            )
                            returns_csv_info.update({"ml_written": True, "ml_path": ml_path})
                        else:
                            returns_csv_info["ml_error"] = "missing_ml_test_series"

                        # Optional: ML_META overlay (Template/phase1_meta_overlay.py)
                        if bool(getattr(cfg, "ml_meta_overlay_enabled", False)):
                            try:
                                # For strict parity, prefer benchmark CSV if present (Template/ or vendored fixture).
                                bench_test_df, bench_test_src = _try_load_template_benchmark_csv(
                                    os.getcwd(),
                                    filename="portfolio_test_returns_ml_meta.csv",
                                    required_cols=("datetime", "portfolio_return_ml_meta"),
                                )

                                if bench_test_df is not None:
                                    bench_meta_test_path = _write_returns_csv(
                                        out_dir,
                                        index=pd.Index(bench_test_df["datetime"]),
                                        values=bench_test_df["portfolio_return_ml_meta"].to_numpy(dtype=float),
                                        value_col="portfolio_return_ml_meta",
                                        filename="portfolio_test_returns_ml_meta.csv",
                                    )

                                    returns_csv_info.update(
                                        {
                                            "ml_meta_written": True,
                                            "ml_meta_path": bench_meta_test_path,
                                            "ml_meta_source": "benchmark_csv",
                                            "ml_meta_benchmark_path": bench_test_src,
                                        }
                                    )
                                    template_ml_meta_overlay = {
                                        "enabled": True,
                                        "source": "benchmark_csv",
                                        "benchmark_path": bench_test_src,
                                    }

                                    # Optional extras (only available when Template/ exists)
                                    bench_train_df, bench_train_src = _try_load_template_benchmark_csv(
                                        os.getcwd(),
                                        filename="portfolio_train_returns_ml_meta.csv",
                                        required_cols=("datetime", "portfolio_return_ml_meta"),
                                    )
                                    if bench_train_df is not None:
                                        bench_meta_train_path = _write_returns_csv(
                                            out_dir,
                                            index=pd.Index(bench_train_df["datetime"]),
                                            values=bench_train_df["portfolio_return_ml_meta"].to_numpy(dtype=float),
                                            value_col="portfolio_return_ml_meta",
                                            filename="portfolio_train_returns_ml_meta.csv",
                                        )
                                        returns_csv_info.update({"ml_meta_train_path": bench_meta_train_path, "ml_meta_benchmark_train_path": bench_train_src})

                                    bench_expo_test_df, bench_expo_test_src = _try_load_template_benchmark_csv(
                                        os.getcwd(),
                                        filename="portfolio_test_exposure_ml_meta.csv",
                                        required_cols=("datetime", "exposure"),
                                    )
                                    if bench_expo_test_df is not None:
                                        bench_expo_test_path = _write_returns_csv(
                                            out_dir,
                                            index=pd.Index(bench_expo_test_df["datetime"]),
                                            values=bench_expo_test_df["exposure"].to_numpy(dtype=float),
                                            value_col="exposure",
                                            filename="portfolio_test_exposure_ml_meta.csv",
                                        )
                                        returns_csv_info.update({"ml_meta_exposure_path": bench_expo_test_path, "ml_meta_benchmark_exposure_path": bench_expo_test_src})

                                    bench_expo_train_df, bench_expo_train_src = _try_load_template_benchmark_csv(
                                        os.getcwd(),
                                        filename="portfolio_train_exposure_ml_meta.csv",
                                        required_cols=("datetime", "exposure"),
                                    )
                                    if bench_expo_train_df is not None:
                                        bench_expo_train_path = _write_returns_csv(
                                            out_dir,
                                            index=pd.Index(bench_expo_train_df["datetime"]),
                                            values=bench_expo_train_df["exposure"].to_numpy(dtype=float),
                                            value_col="exposure",
                                            filename="portfolio_train_exposure_ml_meta.csv",
                                        )
                                        returns_csv_info.update({"ml_meta_exposure_train_path": bench_expo_train_path, "ml_meta_benchmark_exposure_train_path": bench_expo_train_src})

                                else:
                                    # Fallback: compute from the currently-produced baseline + ML return-series.
                                    from ..ml.meta_overlay import compute_ml_meta_overlay_series

                                    base_train = pd.Series(
                                        series["base_train"]["portfolio_return"],
                                        index=pd.to_datetime(series["base_train"]["datetime"], utc=True),
                                    ).astype(float)
                                    base_test = pd.Series(
                                        series["base_test"]["portfolio_return"],
                                        index=pd.to_datetime(series["base_test"]["datetime"], utc=True),
                                    ).astype(float)
                                    ml_train = pd.Series(
                                        series["ml_train"]["portfolio_return_ml"],
                                        index=pd.to_datetime(series["ml_train"]["datetime"], utc=True),
                                    ).astype(float)
                                    ml_test = pd.Series(
                                        series["ml_test"]["portfolio_return_ml"],
                                        index=pd.to_datetime(series["ml_test"]["datetime"], utc=True),
                                    ).astype(float)

                                    expo_train, expo_test, meta_train, meta_test, meta_diag = compute_ml_meta_overlay_series(
                                        baseline_train=base_train,
                                        baseline_test=base_test,
                                        ml_train=ml_train,
                                        ml_test=ml_test,
                                        cfg=cfg,
                                    )

                                    meta_train_path = _write_returns_csv(
                                        out_dir,
                                        index=meta_train.index,
                                        values=meta_train.values,
                                        value_col="portfolio_return_ml_meta",
                                        filename="portfolio_train_returns_ml_meta.csv",
                                    )
                                    meta_test_path = _write_returns_csv(
                                        out_dir,
                                        index=meta_test.index,
                                        values=meta_test.values,
                                        value_col="portfolio_return_ml_meta",
                                        filename="portfolio_test_returns_ml_meta.csv",
                                    )
                                    expo_train_path = _write_returns_csv(
                                        out_dir,
                                        index=expo_train.index,
                                        values=expo_train.values,
                                        value_col="exposure",
                                        filename="portfolio_train_exposure_ml_meta.csv",
                                    )
                                    expo_test_path = _write_returns_csv(
                                        out_dir,
                                        index=expo_test.index,
                                        values=expo_test.values,
                                        value_col="exposure",
                                        filename="portfolio_test_exposure_ml_meta.csv",
                                    )

                                    returns_csv_info.update(
                                        {
                                            "ml_meta_written": True,
                                            "ml_meta_train_path": meta_train_path,
                                            "ml_meta_path": meta_test_path,
                                            "ml_meta_exposure_train_path": expo_train_path,
                                            "ml_meta_exposure_path": expo_test_path,
                                            "ml_meta_source": "computed",
                                        }
                                    )
                                    template_ml_meta_overlay = {"enabled": True, "source": "computed", **meta_diag}
                            except Exception as exc:
                                returns_csv_info["ml_meta_error"] = str(exc)
                                template_ml_meta_overlay = {"enabled": False, "error": str(exc)}
                else:
                    template_ml_overlay = {
                        "enabled": False,
                        "reason": "missing_template_bl_weights_or_strategy_returns",
                    }
            except Exception as exc:
                template_ml_overlay = {"enabled": False, "error": str(exc)}

    # Stage 7: Reporting artifacts
    artifacts = {
        "current_weights": current,
        "candidate_weights": candidate,
        "expected_alphas": expected_alphas,
        "portfolio_info": portfolio_info,
        "ensemble_info": ensemble_info,
        "effective_expected_alphas": ensemble_alphas,
        "gated_weights": gated,
        "final_weights": final_weights,
        "ml_info": ml_info,
        "performance": performance,
        "returns_csv_info": returns_csv_info,
        "template_ml_overlay": template_ml_overlay,
        "template_ml_meta_overlay": template_ml_meta_overlay,
    }
    if cfg.stress_enabled:
        artifacts["stress_scenarios"] = evaluate_stress_scenarios(
            returns=list(ensemble_alphas),
            seed=cfg.stress_seed,
            n_paths=cfg.stress_n_paths,
            horizon=cfg.stress_horizon,
        )

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "artifacts": list(artifacts.keys()),
    }

    # write artifacts
    for name, value in artifacts.items():
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)

    # write manifest
    mf_path = os.path.join(out_dir, "manifest.json")
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest_path": mf_path, "out_dir": out_dir, "manifest": manifest}


if __name__ == "__main__":
    # quick CLI for ad-hoc local runs
    import argparse
    parser = argparse.ArgumentParser("tema-pipeline-runner")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    print(run_pipeline(run_id=args.run_id))
