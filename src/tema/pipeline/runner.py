from typing import List, Sequence, Optional
from ..config import BacktestConfig
from ..turnover import apply_rebalance_gating
from ..ensemble import DynamicEnsembleConfig, combine_stream_signals, compute_dynamic_ensemble_weights
from ..online_learning import OnlineLogisticLearner
from ..stress import evaluate_stress_scenarios
from ..data import load_price_panel, split_train_test
from ..signals import resolve_signal_engine
from ..portfolio import allocate_portfolio_weights
from ..backtest import build_weight_schedule_from_signals, run_return_equity_simulation
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


def _annualization_factor(freq: str) -> float:
    mapping = {
        "D": 252.0,
        "H": 252.0 * 24.0,
        "W": 52.0,
        "M": 12.0,
    }
    return float(mapping.get(str(freq).upper(), 252.0))


def _effective_data_max_assets(cfg: BacktestConfig) -> tuple[Optional[int], bool]:
    max_assets = cfg.data_max_assets
    if (
        cfg.modular_data_signals_enabled
        and cfg.data_full_universe_for_parity
        and int(max_assets) == 3
    ):
        return None, True
    return int(max_assets), False


def _load_data_context(cfg: BacktestConfig) -> dict:
    max_assets, full_universe_override = _effective_data_max_assets(cfg)
    price_df = load_price_panel(
        data_path=cfg.data_path,
        root=os.getcwd(),
        max_assets=max_assets,
        min_rows=max(3, cfg.data_min_rows),
    )
    train_df, test_df = split_train_test(price_df, train_ratio=cfg.data_train_ratio)
    if train_df.empty or test_df.empty:
        raise ValueError("train/test split produced empty partition")
    train_returns = (
        train_df.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .fillna(0.0)
    )
    return {
        "price_df": price_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_returns": train_returns,
        "max_assets_used": max_assets,
        "full_universe_override": full_universe_override,
    }


def _vol_proxy_from_train_window(train_returns_window: Optional[np.ndarray], weights: Sequence[float], freq: str) -> float:
    if train_returns_window is None:
        return 0.10
    returns = np.asarray(train_returns_window, dtype=float)
    w = np.asarray(weights, dtype=float)
    if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] != len(w):
        return 0.10
    pnl = returns @ w
    if pnl.size == 0:
        return 0.10
    std = float(np.std(pnl, ddof=0))
    if std <= 1e-12:
        return 0.10
    return std * float(np.sqrt(_annualization_factor(freq)))


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
        returns_df = (
            test_df.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(how="all")
            .fillna(0.0)
        )
        if returns_df.empty:
            raise ValueError("test returns panel is empty")

        weights_path = _constant_weight_schedule(final_weights, len(returns_df))
        if cfg.modular_data_signals_enabled:
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

        sim = run_return_equity_simulation(
            asset_returns=returns_df.to_numpy(dtype=float),
            target_weights=weights_path,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
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
    if cfg.modular_data_signals_enabled:
        try:
            ctx = data_context if data_context is not None else _load_data_context(cfg)
            price_df = ctx["price_df"]
            train_df = ctx["train_df"]
            test_df = ctx["test_df"]
            train_returns = ctx["train_returns"]
            engine = resolve_signal_engine(use_cpp=cfg.signal_use_cpp, cpp_engine=None)
            signal_df = engine.generate(
                price_df=train_df,
                fast_period=cfg.signal_fast_period,
                slow_period=cfg.signal_slow_period,
                method=cfg.signal_method,
            )
            latest_signal = signal_df.iloc[-1].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            latest_ret = train_df.pct_change(fill_method=None).iloc[-1].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            expected_alphas = (latest_signal * latest_ret).to_numpy(dtype=float)
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
            if cfg.portfolio_modular_enabled:
                alloc = allocate_portfolio_weights(
                    expected_alphas=expected_alphas,
                    returns_window=train_returns.to_numpy(dtype=float),
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
                "portfolio_modular_enabled": bool(cfg.portfolio_modular_enabled),
                "portfolio_method": portfolio_method,
                "portfolio_allocation_fallback_used": portfolio_alloc_fallback,
                "portfolio_diagnostics": portfolio_alloc_diag,
            }, train_returns.to_numpy(dtype=float)
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
    if cfg.vol_target_enabled and cfg.vol_target_apply_to_ml:
        target = max(float(cfg.vol_target_annual), 1e-6)
        realized_vol = _vol_proxy_from_train_window(train_returns_window, normalized, cfg.freq)
        leverage = max(
            cfg.vol_target_min_leverage,
            min(cfg.vol_target_max_leverage, target / max(realized_vol, 1e-6)),
        )
        normalized = [x * leverage for x in normalized]
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


def run_pipeline(run_id: Optional[str] = None, cfg: Optional[BacktestConfig] = None, out_root: str = "outputs") -> dict:
    """Execute Wave 2 simplified pipeline and write artifacts under outputs/{run_id}/.

    Returns a dict summary which is also written to manifest.json.
    """
    if run_id is None:
        run_id = datetime.utcnow().strftime("run-%Y%m%dT%H%M%SZ")
    if cfg is None:
        cfg = BacktestConfig()

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
    if cfg.modular_data_signals_enabled:
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
