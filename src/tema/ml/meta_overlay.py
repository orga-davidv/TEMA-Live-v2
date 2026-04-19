from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tema.config import BacktestConfig


def _make_features(baseline: pd.Series, ml: pd.Series, *, lags: int, roll: int) -> pd.DataFrame:
    df = pd.DataFrame({"baseline": baseline, "ml": ml}).copy()
    df_shift = df.shift(1)

    feat = pd.DataFrame(index=df.index)
    for lag in range(1, int(lags) + 1):
        feat[f"ml_lag_{lag}"] = df["ml"].shift(lag)
        feat[f"base_lag_{lag}"] = df["baseline"].shift(lag)

    feat["ml_roll_mean"] = df_shift["ml"].rolling(int(roll), min_periods=1).mean()
    feat["ml_roll_vol"] = df_shift["ml"].rolling(int(roll), min_periods=1).std().fillna(0.0)
    feat["base_roll_mean"] = df_shift["baseline"].rolling(int(roll), min_periods=1).mean()
    feat["base_roll_vol"] = df_shift["baseline"].rolling(int(roll), min_periods=1).std().fillna(0.0)

    return feat.fillna(0.0)


def _fit_linear_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    xb = np.hstack([X, ones])
    return np.linalg.pinv(xb) @ y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    xb = np.hstack([X, ones])
    return xb @ w


def _exposure_from_signed_proba(proba: np.ndarray, k: float, floor: float) -> np.ndarray:
    signed = 2.0 * (proba - 0.5)
    raw = np.clip(k * signed, -1.0, 1.0)
    return np.where(
        np.abs(raw) > 0.0,
        np.sign(raw) * (floor + (1.0 - floor) * np.abs(raw)),
        0.0,
    )


def _exposure_from_pred(pred: np.ndarray, k: float, floor: float) -> np.ndarray:
    raw = np.clip(k * pred, -1.0, 1.0)
    return np.where(
        np.abs(raw) > 0.0,
        np.sign(raw) * (floor + (1.0 - floor) * np.abs(raw)),
        0.0,
    )


def _compute_metrics_arithmetic(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = returns.fillna(0.0)
    cumulative = (1.0 + r).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0) if len(cumulative) else 0.0
    mean_ret = float(r.mean()) if len(r) else 0.0
    vol = float(r.std()) if len(r) else 0.0
    ann_ret = mean_ret * float(freq)
    ann_vol = vol * math.sqrt(float(freq))
    sharpe = (mean_ret / vol) * math.sqrt(float(freq)) if vol > 0 else 0.0
    running_max = cumulative.cummax() if len(cumulative) else cumulative
    drawdown = (cumulative - running_max) / running_max if len(cumulative) else cumulative
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def _align_features_targets(feat: pd.DataFrame, ml: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    df = feat.copy()
    df["ml"] = ml
    df["y"] = (df["ml"] > 0.0).astype(float)
    df = df.dropna(subset=["ml"])
    y = df["y"].values
    X = df.drop(columns=["ml", "y"]).values
    return X, y, df.index


def _build_time_ordered_validation_folds(
    *,
    n_rows: int,
    min_rows: int,
    validation_ratio: float,
    requested_folds: int,
) -> list[tuple[int, int, int]]:
    n = int(n_rows)
    min_r = max(int(min_rows), 1)
    if n < (2 * min_r):
        return []

    val_rows = max(min_r, int(round(float(n) * float(validation_ratio))))
    if val_rows >= n:
        val_rows = max(min_r, n - min_r)
    if val_rows <= 0:
        return []

    max_folds = max(1, (n - min_r) // val_rows)
    folds = max(1, min(int(requested_folds), max_folds))

    start = n - (folds * val_rows)
    if start < min_r:
        start = min_r
        val_rows = max(min_r, (n - start) // folds)
        if val_rows <= 0:
            return []

    out: list[tuple[int, int, int]] = []
    for fold_idx in range(folds):
        split_idx = start + fold_idx * val_rows
        end_idx = split_idx + val_rows
        if fold_idx == folds - 1:
            end_idx = n
        if split_idx < min_r:
            continue
        if end_idx - split_idx < min_r:
            continue
        out.append((fold_idx, int(split_idx), int(min(end_idx, n))))
    return out


def compute_ml_meta_overlay_series(
    *,
    baseline_train: pd.Series,
    baseline_test: pd.Series,
    ml_train: pd.Series,
    ml_test: pd.Series,
    cfg: BacktestConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    """Reproduce Template/phase1_meta_overlay.py.

    Learns an exposure time-series on train, then applies to test:
      ml_meta = exposure(t) * ml(t)

    Returns:
      (expo_train, expo_test, ml_meta_train, ml_meta_test, diagnostics)
    """

    lags = int(getattr(cfg, "ml_meta_lags", 5))
    roll = int(getattr(cfg, "ml_meta_roll", 5))

    feat_train = _make_features(baseline_train, ml_train, lags=lags, roll=roll)
    feat_test = _make_features(baseline_test, ml_test, lags=lags, roll=roll)

    X_train, y_train, idx_train = _align_features_targets(feat_train, ml_train)
    X_test, _y_dummy, idx_test = _align_features_targets(feat_test, ml_test)

    if X_train.size == 0 or X_test.size == 0:
        zero_train = pd.Series(0.0, index=idx_train, name="exposure")
        zero_test = pd.Series(0.0, index=idx_test, name="exposure")
        train_meta = pd.Series(0.0, index=idx_train, name="portfolio_return_ml_meta")
        test_meta = pd.Series(0.0, index=idx_test, name="portfolio_return_ml_meta")
        return zero_train, zero_test, train_meta, test_meta, {"enabled": False, "reason": "empty_features"}

    ks = np.linspace(
        float(getattr(cfg, "ml_meta_k_min", 0.5)),
        float(getattr(cfg, "ml_meta_k_max", 8.0)),
        int(getattr(cfg, "ml_meta_k_steps", 32)),
    )
    floors = list(getattr(cfg, "ml_meta_floors", (0.2, 0.4, 0.6, 0.8, 0.9)))
    computed_mode = not bool(getattr(cfg, "template_use_precomputed_artifacts", True))
    if computed_mode and bool(getattr(cfg, "ml_meta_computed_allow_zero_floor", True)):
        floors = sorted({*floors, 0.0})

    min_vol_ratio = float(getattr(cfg, "ml_meta_min_vol_ratio", 0.5))
    min_mean_abs_exposure = float(getattr(cfg, "ml_meta_min_mean_abs_exposure", 0.7))
    min_turnover_per_year = float(getattr(cfg, "ml_meta_min_turnover_per_year", 5.0))
    target_mean_abs_exposure = float(getattr(cfg, "ml_meta_target_mean_abs_exposure", 0.9))
    target_turnover_per_year = float(getattr(cfg, "ml_meta_target_turnover_per_year", 20.0))
    if computed_mode:
        min_vol_ratio = 0.0
        min_mean_abs_exposure = 0.0
        min_turnover_per_year = 0.0
        target_mean_abs_exposure = float(getattr(cfg, "ml_meta_computed_target_mean_abs_exposure", 0.10))
        target_turnover_per_year = float(getattr(cfg, "ml_meta_computed_target_turnover_per_year", 2.0))
        computed_exposure_tol = float(getattr(cfg, "ml_meta_computed_exposure_tolerance", 0.10))
        computed_turnover_tol = float(getattr(cfg, "ml_meta_computed_turnover_tolerance", 2.0))

    train_ml_metrics = _compute_metrics_arithmetic(ml_train.loc[idx_train])
    train_ml_ann_vol = float(train_ml_metrics["annualized_vol"])

    def evaluate_candidate(expo: np.ndarray, series_vals: np.ndarray) -> Dict[str, float]:
        r = pd.Series(series_vals * expo)
        n = len(r)
        mean = float(r.mean()) if n else 0.0
        std = float(r.std()) if n else 0.0
        ann_vol = std * math.sqrt(252.0) if std > 0 else 0.0
        ann_ret = mean * 252.0
        sharpe = (mean / std) * math.sqrt(252.0) if std > 0 else 0.0
        turnover_per_year = float(np.mean(np.abs(np.diff(expo)))) * 252.0 if n > 1 else 0.0
        mean_abs_exposure = float(np.mean(np.abs(expo))) if n else 0.0
        total_return = float((1.0 + r).cumprod().iloc[-1] - 1.0) if n else 0.0
        return {
            "mean": float(mean),
            "std": float(std),
            "annualized_vol": float(ann_vol),
            "annualized_return": float(ann_ret),
            "sharpe": float(sharpe),
            "turnover_per_year": float(turnover_per_year),
            "mean_abs_exposure": float(mean_abs_exposure),
            "total_return": float(total_return),
        }

    def candidate_utility(cand: Dict[str, float], floor_value: float) -> float:
        if computed_mode:
            exposure_shape = math.exp(
                -abs(cand["mean_abs_exposure"] - target_mean_abs_exposure) / max(computed_exposure_tol, 1e-9)
            )
            turnover_shape = math.exp(
                -abs(cand["turnover_per_year"] - target_turnover_per_year) / max(computed_turnover_tol, 1e-9)
            )
            return float(cand["sharpe"] * exposure_shape * turnover_shape)

        exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
        turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
        util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
        return float(util) + (1e-6 if float(floor_value) == 0.0 else 0.0)

    def candidate_is_eligible(cand: Dict[str, float], ann_vol_ref: float) -> bool:
        if cand["annualized_vol"] < min_vol_ratio * ann_vol_ref:
            return False
        if cand["total_return"] <= 0.0:
            return False
        if cand["mean_abs_exposure"] < min_mean_abs_exposure:
            return False
        if cand["turnover_per_year"] < min_turnover_per_year:
            return False
        return True

    y_cont = ml_train.loc[idx_train].values
    best: Dict[str, float | str] = {"utility": -1e9}
    candidate_specs = [
        ("signed_proba", float(k), float(f))
        for k in ks
        for f in floors
    ] + [
        ("pred", float(k), float(f))
        for k in ks
        for f in floors
    ]

    if computed_mode:
        val_ratio = float(np.clip(getattr(cfg, "ml_meta_computed_validation_ratio", 0.25), 0.10, 0.50))
        min_rows = max(int(getattr(cfg, "ml_meta_computed_validation_min_rows", 80)), 20)
        folds_requested = int(getattr(cfg, "ml_meta_computed_validation_folds", 3))
        overfit_penalty = float(max(getattr(cfg, "ml_meta_computed_overfit_penalty", 0.5), 0.0))
        fold_slices = _build_time_ordered_validation_folds(
            n_rows=len(idx_train),
            min_rows=min_rows,
            validation_ratio=val_ratio,
            requested_folds=folds_requested,
        )

        if fold_slices:
            agg: dict[tuple[str, float, float], dict[str, list[float]]] = {
                spec: {"scores": [], "val_utility": [], "subtrain_utility": []}
                for spec in candidate_specs
            }

            for _fold_idx, split_idx, end_idx in fold_slices:
                X_sub = X_train[:split_idx]
                X_val = X_train[split_idx:end_idx]
                y_sub = y_train[:split_idx]
                if X_sub.size == 0 or X_val.size == 0:
                    continue

                w_prob_fold = _fit_linear_pinv(X_sub, y_sub)
                proba_sub = _sigmoid(_predict(X_sub, w_prob_fold))
                proba_val = _sigmoid(_predict(X_val, w_prob_fold))

                y_cont_sub = y_cont[:split_idx]
                y_cont_val = y_cont[split_idx:end_idx]
                w_ret_fold = _fit_linear_pinv(X_sub, y_cont_sub)
                pred_sub = _predict(X_sub, w_ret_fold)
                pred_val = _predict(X_val, w_ret_fold)

                sub_ml_ann_vol = float(_compute_metrics_arithmetic(pd.Series(y_cont_sub))["annualized_vol"])
                for method, k, f in candidate_specs:
                    if method == "signed_proba":
                        expo_sub = _exposure_from_signed_proba(proba_sub, k, f)
                        expo_val = _exposure_from_signed_proba(proba_val, k, f)
                    else:
                        expo_sub = _exposure_from_pred(pred_sub, k, f)
                        expo_val = _exposure_from_pred(pred_val, k, f)

                    cand_sub = evaluate_candidate(expo_sub, y_cont_sub)
                    cand_val = evaluate_candidate(expo_val, y_cont_val)
                    if not candidate_is_eligible(cand_val, sub_ml_ann_vol):
                        continue

                    util_sub = candidate_utility(cand_sub, f)
                    util_val = candidate_utility(cand_val, f)
                    score = float(util_val - overfit_penalty * abs(util_sub - util_val))
                    stats = agg[(method, k, f)]
                    stats["scores"].append(score)
                    stats["val_utility"].append(float(util_val))
                    stats["subtrain_utility"].append(float(util_sub))

            ranked: list[dict[str, float | str]] = []
            for (method, k, f), stats in agg.items():
                if not stats["scores"]:
                    continue
                ranked.append(
                    {
                        "utility": float(np.mean(stats["scores"])),
                        "val_utility": float(np.mean(stats["val_utility"])),
                        "subtrain_utility": float(np.mean(stats["subtrain_utility"])),
                        "utility_std": float(np.std(stats["scores"])),
                        "overfit_gap": float(
                            np.mean(np.abs(np.asarray(stats["subtrain_utility"]) - np.asarray(stats["val_utility"])))
                        ),
                        "method": method,
                        "k": float(k),
                        "f": float(f),
                        "folds_evaluated": int(len(stats["scores"])),
                    }
                )

            if ranked:
                best = max(
                    ranked,
                    key=lambda item: (
                        float(item.get("utility", float("-inf"))),
                        float(item.get("val_utility", float("-inf"))),
                        -float(item.get("utility_std", float("inf"))),
                        -float(item.get("overfit_gap", float("inf"))),
                        -float(item.get("f", 0.0)),
                        -float(item.get("k", 0.0)),
                        1.0 if str(item.get("method", "")) == "pred" else 0.0,
                    ),
                )

    if float(best["utility"]) < 0:
        w_prob = _fit_linear_pinv(X_train, y_train)
        proba_train = _sigmoid(_predict(X_train, w_prob))
        w_ret = _fit_linear_pinv(X_train, y_cont)
        pred_train = _predict(X_train, w_ret)
        series_vals = ml_train.loc[idx_train].values

        for method, k, f in candidate_specs:
            expo = _exposure_from_signed_proba(proba_train, k, f) if method == "signed_proba" else _exposure_from_pred(pred_train, k, f)
            cand = evaluate_candidate(expo, series_vals)
            if not candidate_is_eligible(cand, train_ml_ann_vol):
                continue
            util = candidate_utility(cand, f)
            if util > float(best["utility"]):
                best = {"utility": float(util), "method": method, "k": float(k), "f": float(f), **cand}

    if float(best["utility"]) < 0:
        w_prob = _fit_linear_pinv(X_train, y_train)
        proba_train = _sigmoid(_predict(X_train, w_prob))
        w_ret = _fit_linear_pinv(X_train, y_cont)
        pred_train = _predict(X_train, w_ret)
        candidate_best_util = -1e9
        candidate_best = None
        preferred_floors = {0.0, 0.05, 0.1, 0.2} if computed_mode else {0.6, 0.8, 0.9}
        series_vals = ml_train.loc[idx_train].values
        for method, k, f in candidate_specs:
            expo = _exposure_from_pred(pred_train, float(k), float(f)) if method == "pred" else _exposure_from_signed_proba(proba_train, float(k), float(f))
            cand = evaluate_candidate(expo, series_vals)
            util = candidate_utility(cand, float(f))
            score = float(util) + (1e-3 if float(f) in preferred_floors else 0.0)
            if score > candidate_best_util:
                candidate_best_util = score
                candidate_best = {"utility": float(util), "method": method, "k": float(k), "f": float(f), **cand}
        if candidate_best is not None:
            best = candidate_best

    method = str(best.get("method", "pred"))
    k_best = float(best.get("k", 1.0))
    f_best = float(best.get("f", 0.0))

    w_prob = _fit_linear_pinv(X_train, y_train)
    proba_train = _sigmoid(_predict(X_train, w_prob))
    proba_test = _sigmoid(_predict(X_test, w_prob))
    w_ret = _fit_linear_pinv(X_train, y_cont)
    pred_train = _predict(X_train, w_ret)
    pred_test = _predict(X_test, w_ret)

    if method == "signed_proba":
        expo_train = _exposure_from_signed_proba(proba_train, k_best, f_best)
        expo_test = _exposure_from_signed_proba(proba_test, k_best, f_best)
    else:
        expo_train = _exposure_from_pred(pred_train, k_best, f_best)
        expo_test = _exposure_from_pred(pred_test, k_best, f_best)

    expo_train_s = pd.Series(expo_train, index=idx_train, name="exposure")
    expo_test_s = pd.Series(expo_test, index=idx_test, name="exposure")

    ml_meta_train = pd.Series(expo_train * ml_train.loc[idx_train].values, index=idx_train, name="portfolio_return_ml_meta")
    ml_meta_test = pd.Series(expo_test * ml_test.loc[idx_test].values, index=idx_test, name="portfolio_return_ml_meta")

    diag = {
        "enabled": True,
        "chosen_method": method,
        "chosen_k": float(k_best),
        "chosen_floor": float(f_best),
        "train_ml_ann_vol": float(train_ml_ann_vol),
        "selection_mode": "computed_regularized" if computed_mode else "template_compatible",
        "selection": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best.items()},
    }
    if computed_mode:
        diag["computed_validation_ratio"] = float(np.clip(getattr(cfg, "ml_meta_computed_validation_ratio", 0.25), 0.10, 0.50))
        diag["computed_validation_folds"] = float(max(int(getattr(cfg, "ml_meta_computed_validation_folds", 3)), 1))
        diag["computed_overfit_penalty"] = float(max(getattr(cfg, "ml_meta_computed_overfit_penalty", 0.5), 0.0))

    return expo_train_s, expo_test_s, ml_meta_train, ml_meta_test, diag
