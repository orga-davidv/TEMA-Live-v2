from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tema.backtest import compute_backtest_metrics
from tema.config import BacktestConfig

from .cpp_hmm import get_hmm_engine


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


def evaluate_weighted_portfolio_returns(returns_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = returns_df.reindex(columns=weights.index).fillna(0.0)
    return aligned.mul(weights, axis=1).sum(axis=1)


def build_rf_feature_matrix(returns: pd.Series, hmm_probs: np.ndarray) -> pd.DataFrame:
    r = returns.fillna(0.0)
    f = pd.DataFrame(index=r.index)
    f["ret_1"] = r.shift(1).fillna(0.0)
    f["ret_5"] = r.rolling(5, min_periods=1).mean().shift(1).fillna(0.0)
    f["ret_20"] = r.rolling(20, min_periods=1).mean().shift(1).fillna(0.0)
    f["vol_20"] = r.rolling(20, min_periods=2).std(ddof=0).shift(1).fillna(0.0)
    f["abs_ret_1"] = r.abs().shift(1).fillna(0.0)

    for k in range(int(hmm_probs.shape[1])):
        f[f"hmm_p_{k}"] = hmm_probs[:, k]

    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def calibrate_soft_threshold_for_target_exposure(
    *,
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
    *,
    raw_scale: np.ndarray,
    probs: np.ndarray,
    returns: pd.Series,
    cfg: BacktestConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
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
    *,
    train_port_rets: pd.Series,
    test_port_rets: pd.Series,
    cfg: BacktestConfig,
    hmm_engine,
) -> Tuple[pd.Series, pd.Series, Dict[str, float], pd.DataFrame]:
    tr = train_port_rets.fillna(0.0)
    te = test_port_rets.fillna(0.0)

    train_probs, test_probs, means, variances = hmm_engine.fit_forward_probs(
        train_returns=tr.to_numpy(dtype=np.float64),
        test_returns=te.to_numpy(dtype=np.float64),
        n_states=int(cfg.hmm_n_states),
        n_iter=int(cfg.hmm_n_iter),
        var_floor=float(cfg.hmm_var_floor),
        trans_sticky=float(cfg.hmm_trans_sticky),
    )

    x_train = build_rf_feature_matrix(tr, train_probs)
    x_test = build_rf_feature_matrix(te, test_probs)

    y_train = (tr.shift(-1) > 0.0).astype(int)
    x_train = x_train.iloc[:-1]
    y_train = y_train.iloc[:-1]

    clf = RandomForestClassifier(
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=int(cfg.rf_max_depth),
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
        random_state=int(cfg.rf_random_state),
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    p_train = clf.predict_proba(build_rf_feature_matrix(tr, train_probs))[:, 1]
    p_test = clf.predict_proba(x_test)[:, 1]

    threshold = float(cfg.ml_prob_threshold)
    if bool(cfg.ml_auto_threshold):
        threshold = calibrate_soft_threshold_for_target_exposure(
            probs=p_train,
            target_exposure=float(cfg.ml_target_exposure),
            fallback_threshold=threshold,
        )

    denom = max(1.0 - threshold, 1e-9)
    scale_train_raw = np.clip((p_train - threshold) / denom, 0.0, 1.0)
    scale_test_raw = np.clip((p_test - threshold) / denom, 0.0, 1.0)

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
        "ml_threshold_used": float(threshold),
        "ml_auto_threshold": float(1.0 if cfg.ml_auto_threshold else 0.0),
        "ml_target_exposure": float(cfg.ml_target_exposure),
        "train_avg_exposure": float(np.mean(scale_train)),
        "test_avg_exposure": float(np.mean(scale_test)),
        "train_avg_exposure_raw": float(np.mean(scale_train_raw)),
        "test_avg_exposure_raw": float(np.mean(scale_test_raw)),
        "rebalance_min_threshold": float(getattr(cfg, "rebalance_min_threshold", 0.0)),
        "cost_aware_rebalance": float(1.0 if bool(getattr(cfg, "cost_aware_rebalance", False)) else 0.0),
        "cost_aware_rebalance_multiplier": float(getattr(cfg, "cost_aware_rebalance_multiplier", 1.0)),
        "cost_aware_alpha_lookback": float(getattr(cfg, "cost_aware_alpha_lookback", 20)),
    }
    diag.update({f"gate_{k}": float(v) for k, v in gate_diag.items()})

    hmm_df = pd.DataFrame({"state": np.arange(int(cfg.hmm_n_states), dtype=int), "mean_return": means, "variance": variances})
    return ml_train, ml_test, diag, hmm_df


def compute_and_apply_ml_position_scalar(
    *,
    cfg: BacktestConfig,
    ml_train_rets: pd.Series,
    ml_test_rets: pd.Series,
    train_port_rets: Optional[pd.Series] = None,
    test_port_rets: Optional[pd.Series] = None,
    hmm_engine=None,
) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    diag: Dict[str, object] = {}

    if ml_train_rets is None or ml_train_rets.empty:
        diag["enabled"] = False
        diag["auto"] = bool(getattr(cfg, "ml_position_scalar_auto", True))
        return ml_train_rets, ml_test_rets, diag

    target_vol = float(getattr(cfg, "ml_position_scalar_target_vol", 0.10))
    max_scalar = float(getattr(cfg, "ml_position_scalar_max", 50.0))
    hmm_floor = float(getattr(cfg, "ml_hmm_scalar_floor", 0.30))
    hmm_ceiling = float(getattr(cfg, "ml_hmm_scalar_ceiling", 1.50))

    method = getattr(cfg, "ml_position_scalar_method", "hmm_prob")

    def _ann_vol(series: pd.Series) -> float:
        return float(series.std(ddof=0) * np.sqrt(252.0)) if (series is not None and not series.empty) else 0.0

    def _shape_hmm_raw(raw: pd.Series) -> pd.Series:
        lo = min(hmm_floor, hmm_ceiling)
        hi = max(hmm_floor, hmm_ceiling)
        clipped = raw.clip(lower=0.0, upper=1.0)
        return (lo + (hi - lo) * clipped).astype(float)

    if method == "hmm_prob" and train_port_rets is not None and test_port_rets is not None:
        n_states = int(getattr(cfg, "hmm_n_states", 3))

        def _calibrate_from_raw(
            raw_train: pd.Series,
            raw_test: pd.Series,
            method_name: str,
            bull_state: int,
        ) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
            pre_vol = _ann_vol(ml_train_rets * raw_train)
            factor = 0.0 if (pre_vol <= 0 or not np.isfinite(pre_vol)) else (target_vol / pre_vol)

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

            return _calibrate_from_raw(_shape_hmm_raw(raw_train), _shape_hmm_raw(raw_test), "hmm_prob_python", bull_state)
        except Exception as exc:
            diag["python_hmm_error"] = str(exc)

        if hmm_engine is not None:
            try:
                tr = train_port_rets.fillna(0.0).to_numpy(dtype=np.float64)
                te = test_port_rets.fillna(0.0).to_numpy(dtype=np.float64)

                train_probs, test_probs, means, _variances = hmm_engine.fit_forward_probs(
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

                return _calibrate_from_raw(_shape_hmm_raw(raw_train), _shape_hmm_raw(raw_test), "hmm_prob_cpp", bull_state)
            except Exception as exc:
                diag["hmm_error"] = str(exc)

    train_ml_vol = float(ml_train_rets.std(ddof=0) * np.sqrt(252.0))
    unclipped_scalar = None
    clipped = False

    if bool(getattr(cfg, "ml_position_scalar_auto", True)):
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
        **diag,
    }
    return scaled_ml_train, scaled_ml_test, diag


def compute_and_apply_vol_target(
    *,
    cfg: BacktestConfig,
    train_port_rets: pd.Series,
    test_port_rets: pd.Series,
    ml_train_rets: pd.Series,
    ml_test_rets: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, object]]:
    diag: Dict[str, object] = {}
    if not bool(getattr(cfg, "vol_target_enabled", False)):
        diag["enabled"] = False
        return train_port_rets, test_port_rets, ml_train_rets, ml_test_rets, diag

    ref_raw = getattr(cfg, "vol_target_reference", "ml")
    ref = str(ref_raw).lower() if ref_raw is not None else "ml"
    fallback_used = False

    if ref == "ml":
        if ml_train_rets is not None and not ml_train_rets.empty:
            reference_series = ml_train_rets
        else:
            reference_series = train_port_rets
            fallback_used = True
    elif ref == "bl":
        reference_series = train_port_rets
    else:
        reference_series = train_port_rets
        fallback_used = True

    reference_train_vol = float(reference_series.std(ddof=0) * np.sqrt(252.0)) if not reference_series.empty else 0.0
    target_vol = float(cfg.vol_target_annual)

    if reference_train_vol <= 1e-12 or not np.isfinite(reference_train_vol):
        unclipped_scalar = float(cfg.vol_target_max_leverage)
        scalar = float(cfg.vol_target_max_leverage)
        clipped = True
    else:
        unclipped_scalar = float(target_vol / reference_train_vol)
        scalar = float(np.clip(unclipped_scalar, cfg.vol_target_min_leverage, cfg.vol_target_max_leverage))
        clipped = not np.isclose(unclipped_scalar, scalar)

    scaled_train = train_port_rets * scalar
    scaled_test = test_port_rets * scalar

    apply_to_ml = bool(getattr(cfg, "vol_target_apply_to_ml", False))
    if ml_train_rets is not None and not ml_train_rets.empty:
        if apply_to_ml:
            scaled_ml_train = ml_train_rets * scalar
            scaled_ml_test = ml_test_rets * scalar
        else:
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


def compute_template_ml_overlay(
    *,
    train_returns_df: pd.DataFrame,
    test_returns_df: pd.DataFrame,
    weights: pd.Series,
    cfg: BacktestConfig,
    hmm_engine=None,
    include_series: bool = False,
) -> dict:
    train_port_rets = evaluate_weighted_portfolio_returns(train_returns_df, weights)
    test_port_rets = evaluate_weighted_portfolio_returns(test_returns_df, weights)

    ml_train_rets = train_port_rets.copy()
    ml_test_rets = test_port_rets.copy()
    ml_diag: Dict[str, float] = {}
    hmm_state_df = pd.DataFrame()

    selected_cfg = cfg
    if bool(getattr(cfg, "ml_enabled", False)):
        hmm_engine = hmm_engine or get_hmm_engine(prefer_cpp=True)
        ml_train_rets, ml_test_rets, ml_diag, hmm_state_df = apply_hmm_softprob_rf_strategy(
            train_port_rets=train_port_rets,
            test_port_rets=test_port_rets,
            cfg=selected_cfg,
            hmm_engine=hmm_engine,
        )

    ml_train_rets, ml_test_rets, ml_pos_diag = compute_and_apply_ml_position_scalar(
        cfg=selected_cfg,
        ml_train_rets=ml_train_rets,
        ml_test_rets=ml_test_rets,
        train_port_rets=train_port_rets,
        test_port_rets=test_port_rets,
        hmm_engine=hmm_engine,
    )

    scaled_train, scaled_test, scaled_ml_train, scaled_ml_test, vol_diag = compute_and_apply_vol_target(
        cfg=cfg,
        train_port_rets=train_port_rets,
        test_port_rets=test_port_rets,
        ml_train_rets=ml_train_rets,
        ml_test_rets=ml_test_rets,
    )

    ann = periods_per_year(cfg.freq)

    def _metrics(series: pd.Series) -> dict:
        r = series.fillna(0.0).to_numpy(dtype=float)
        eq = np.cumprod(1.0 + r)
        metrics = compute_backtest_metrics(r, eq, np.zeros_like(r), float(ann))
        metrics["equity_final"] = float(eq[-1]) if eq.size else 1.0
        metrics["total_return"] = float(eq[-1] - 1.0) if eq.size else 0.0
        return metrics

    base_train_metrics = _metrics(scaled_train)
    base_test_metrics = _metrics(scaled_test)
    ml_train_metrics = _metrics(scaled_ml_train)
    ml_test_metrics = _metrics(scaled_ml_test)

    out = {
        "base_train_metrics": base_train_metrics,
        "base_test_metrics": base_test_metrics,
        "ml_train_metrics": ml_train_metrics,
        "ml_test_metrics": ml_test_metrics,
        "ml_diagnostics": ml_diag,
        "ml_position_scalar_diagnostics": ml_pos_diag,
        "vol_target_diagnostics": vol_diag,
        "hmm_state_params": hmm_state_df.to_dict(orient="list") if not hmm_state_df.empty else {},
    }

    if include_series:
        out["series"] = {
            "base_test": {
                "datetime": scaled_test.index.astype(str).tolist(),
                "portfolio_return": scaled_test.astype(float).tolist(),
            },
            "ml_test": {
                "datetime": scaled_ml_test.index.astype(str).tolist(),
                "portfolio_return_ml": scaled_ml_test.astype(float).tolist(),
            },
        }

    return out
