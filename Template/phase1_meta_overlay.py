#!/usr/bin/env python3
"""
Phase1 meta overlay: scale ML series directly (ml_meta = exposure * ml)
Calibrate exposure on train only to maximize train Sharpe of ml_meta.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_series(path: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[0])
    df = df.set_index(df.columns[0])
    s = df[value_col].astype(float)
    s.index = pd.to_datetime(s.index)
    return s


def make_features(baseline: pd.Series, ml: pd.Series, lags: int = 5, roll: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"baseline": baseline, "ml": ml}).copy()
    df_shift = df.shift(1)

    feat = pd.DataFrame(index=df.index)
    for lag in range(1, lags + 1):
        feat[f"ml_lag_{lag}"] = df["ml"].shift(lag)
        feat[f"base_lag_{lag}"] = df["baseline"].shift(lag)

    feat["ml_roll_mean"] = df_shift["ml"].rolling(roll, min_periods=1).mean()
    feat["ml_roll_vol"] = df_shift["ml"].rolling(roll, min_periods=1).std().fillna(0)
    feat["base_roll_mean"] = df_shift["baseline"].rolling(roll, min_periods=1).mean()
    feat["base_roll_vol"] = df_shift["baseline"].rolling(roll, min_periods=1).std().fillna(0)
    return feat.fillna(0.0)


def fit_linear_pinv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    xb = np.hstack([X, ones])
    return np.linalg.pinv(xb) @ y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    xb = np.hstack([X, ones])
    return xb @ w


def exposure_from_signed_proba(proba: np.ndarray, k: float, floor: float) -> np.ndarray:
    signed = 2.0 * (proba - 0.5)
    raw = np.clip(k * signed, -1.0, 1.0)
    return np.where(
        np.abs(raw) > 0.0,
        np.sign(raw) * (floor + (1.0 - floor) * np.abs(raw)),
        0.0,
    )


def exposure_from_pred(pred: np.ndarray, k: float, floor: float) -> np.ndarray:
    raw = np.clip(k * pred, -1.0, 1.0)
    return np.where(
        np.abs(raw) > 0.0,
        np.sign(raw) * (floor + (1.0 - floor) * np.abs(raw)),
        0.0,
    )


def compute_metrics(returns: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = returns.fillna(0.0)
    cumulative = (1.0 + r).cumprod()
    total_return = cumulative.iloc[-1] - 1.0
    mean_ret = r.mean()
    vol = r.std()
    ann_ret = mean_ret * freq
    ann_vol = vol * math.sqrt(freq)
    sharpe = (mean_ret / vol) * math.sqrt(freq) if vol > 0 else 0.0
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def align_features_targets(feat: pd.DataFrame, ml: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    df = feat.copy()
    df["ml"] = ml
    df["y"] = (df["ml"] > 0).astype(float)
    df = df.dropna(subset=["ml"])
    y = df["y"].values
    X = df.drop(columns=["ml", "y"]).values
    return X, y, df.index


def main() -> None:
    base = os.path.dirname(__file__)
    baseline_train = load_series(os.path.join(base, "portfolio_train_returns.csv"), "portfolio_return")
    baseline_test = load_series(os.path.join(base, "portfolio_test_returns.csv"), "portfolio_return")
    ml_train = load_series(os.path.join(base, "portfolio_train_returns_ml.csv"), "portfolio_return_ml")
    ml_test = load_series(os.path.join(base, "portfolio_test_returns_ml.csv"), "portfolio_return_ml")

    feat_train = make_features(baseline_train, ml_train)
    feat_test = make_features(baseline_test, ml_test)
    X_train, y_train, idx_train = align_features_targets(feat_train, ml_train)
    X_test, _, idx_test = align_features_targets(feat_test, ml_test)

    w_prob = fit_linear_pinv(X_train, y_train)
    proba_train = sigmoid(predict(X_train, w_prob))
    proba_test = sigmoid(predict(X_test, w_prob))

    y_cont = ml_train.loc[idx_train].values
    w_ret = fit_linear_pinv(X_train, y_cont)
    pred_train = predict(X_train, w_ret)
    pred_test = predict(X_test, w_ret)

    # Selection grid and constraints for soft-weighting based on utility + hard guards
    best: Dict[str, float | str] = {"utility": -1e9}
    ks = np.linspace(0.5, 8.0, 32)
    # only use meaningful floors to avoid near-flat behavior
    floors = [0.2, 0.4, 0.6, 0.8, 0.9]

    # Constraints / utility defaults
    min_vol_ratio = 0.5  # require ml_meta ann vol >= this * train ml ann vol
    min_mean_abs_exposure = 0.7
    min_turnover_per_year = 5.0
    target_mean_abs_exposure = 0.9
    target_turnover_per_year = 20.0

    # precompute train ML annualized vol for vol floor
    train_ml_metrics = compute_metrics(ml_train.loc[idx_train])
    train_ml_ann_vol = train_ml_metrics["annualized_vol"]

    def evaluate_candidate(expo: np.ndarray, series_vals: np.ndarray) -> Dict[str, float]:
        # metrics for candidate on train
        r = pd.Series(series_vals * expo)
        n = len(r)
        mean = float(r.mean())
        std = float(r.std())
        ann_vol = std * math.sqrt(252.0) if std > 0 else 0.0
        ann_ret = mean * 252.0
        sharpe = (mean / std) * math.sqrt(252.0) if std > 0 else 0.0
        # Turnover/activity proxies for continuous exposure:
        # - turnover_per_year: annualized mean absolute exposure change
        # - mean_abs_exposure: average absolute market participation
        turnover_per_year = float(np.mean(np.abs(np.diff(expo)))) * 252.0 if n > 1 else 0.0
        mean_abs_exposure = float(np.mean(np.abs(expo)))
        total_return = (1.0 + r).cumprod().iloc[-1] - 1.0
        return {
            "mean": mean,
            "std": std,
            "annualized_vol": ann_vol,
            "annualized_return": ann_ret,
            "sharpe": sharpe,
            "turnover_per_year": turnover_per_year,
            "mean_abs_exposure": mean_abs_exposure,
            "total_return": float(total_return),
        }

    for k in ks:
        for f in floors:
            # signed proba
            expo = exposure_from_signed_proba(proba_train, float(k), float(f))
            cand = evaluate_candidate(expo, ml_train.loc[idx_train].values)
            # hard guards
            if cand["annualized_vol"] < min_vol_ratio * train_ml_ann_vol:
                continue
            if cand["total_return"] <= 0.0:
                continue
            if cand["mean_abs_exposure"] < min_mean_abs_exposure:
                continue
            if cand["turnover_per_year"] < min_turnover_per_year:
                continue
            # utility: Sharpe with mild rewards for robust participation/activity
            exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
            turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
            util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
            if util > float(best["utility"]):
                best = {"utility": util, "method": "signed_proba", "k": float(k), "f": float(f), **cand}

    for k in ks:
        for f in floors:
            expo = exposure_from_pred(pred_train, float(k), float(f))
            cand = evaluate_candidate(expo, ml_train.loc[idx_train].values)
            if cand["annualized_vol"] < min_vol_ratio * train_ml_ann_vol:
                continue
            if cand["total_return"] <= 0.0:
                continue
            if cand["mean_abs_exposure"] < min_mean_abs_exposure:
                continue
            if cand["turnover_per_year"] < min_turnover_per_year:
                continue
            exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
            turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
            util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
            if util > float(best["utility"]):
                best = {"utility": util, "method": "pred", "k": float(k), "f": float(f), **cand}

    # Fall back: if no candidate passes guards, keep soft behavior by preferring higher floors.
    if float(best["utility"]) < 0:
        candidate_best_util = -1e9
        candidate_best = None
        preferred_floors = [0.6, 0.8, 0.9]
        for k in ks:
            for f in floors:
                for method in ("pred", "signed_proba"):
                    expo = exposure_from_pred(pred_train, float(k), float(f)) if method == "pred" else exposure_from_signed_proba(proba_train, float(k), float(f))
                    cand = evaluate_candidate(expo, ml_train.loc[idx_train].values)
                    exposure_factor = min(1.0, cand["mean_abs_exposure"] / target_mean_abs_exposure)
                    turnover_factor = min(1.0, cand["turnover_per_year"] / target_turnover_per_year)
                    util = cand["sharpe"] * (0.5 + 0.5 * exposure_factor) * (0.5 + 0.5 * turnover_factor)
                    score = util
                    # prefer higher floor to avoid near-zero exposure
                    if f in preferred_floors:
                        score += 1e-3
                    if score > candidate_best_util:
                        candidate_best_util = score
                        candidate_best = {"utility": util, "method": method, "k": float(k), "f": float(f), **cand}
        if candidate_best is not None:
            best = candidate_best

    # build final exposures using chosen params
    if best.get("method") == "signed_proba":
        expo_train = exposure_from_signed_proba(proba_train, float(best["k"]), float(best["f"]))
        expo_test = exposure_from_signed_proba(proba_test, float(best["k"]), float(best["f"]))
    else:
        expo_train = exposure_from_pred(pred_train, float(best.get("k", 1.0)), float(best.get("f", 0.0)))
        expo_test = exposure_from_pred(pred_test, float(best.get("k", 1.0)), float(best.get("f", 0.0)))

    ml_meta_train = pd.Series(expo_train * ml_train.loc[idx_train].values, index=idx_train, name="portfolio_return_ml_meta")
    ml_meta_test = pd.Series(expo_test * ml_test.loc[idx_test].values, index=idx_test, name="portfolio_return_ml_meta")

    # save outputs (returns)
    ml_meta_train.to_csv(os.path.join(base, "portfolio_train_returns_ml_meta.csv"), header=True)
    ml_meta_test.to_csv(os.path.join(base, "portfolio_test_returns_ml_meta.csv"), header=True)

    # save exposures for train and test
    pd.Series(expo_train, index=idx_train, name="exposure").to_csv(os.path.join(base, "portfolio_train_exposure_ml_meta.csv"), header=True)
    pd.Series(expo_test, index=idx_test, name="exposure").to_csv(os.path.join(base, "portfolio_test_exposure_ml_meta.csv"), header=True)

    # compute combined metrics CSV (as before)
    baseline_all = pd.concat([baseline_train, baseline_test]).sort_index()
    ml_all = pd.concat([ml_train, ml_test]).sort_index()
    ml_meta_all = pd.concat([ml_meta_train, ml_meta_test]).sort_index()

    metrics = []
    for name, series in [("baseline", baseline_all), ("ml", ml_all), ("ml_meta", ml_meta_all)]:
        row = compute_metrics(series)
        row["series"] = name
        metrics.append(row)
    pd.DataFrame(metrics).set_index("series").to_csv(os.path.join(base, "ml_meta_overlay_metrics.csv"))

    # diagnostics
    ml_test_metrics = compute_metrics(ml_test.loc[idx_test])
    ml_meta_test_metrics = compute_metrics(ml_meta_test)
    ml_meta_train_metrics = compute_metrics(ml_meta_train)

    # exposure/order statistics helpers
    def exposure_stats(expo_series: pd.Series) -> Dict[str, float]:
        arr = expo_series.fillna(0.0).values
        abs_expo = np.abs(arr)
        diffs = np.abs(np.diff(arr)) if len(arr) > 1 else np.array([])
        mean_abs_exposure = float(np.mean(abs_expo)) if len(abs_expo) > 0 else 0.0
        mean_order_size = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        median_order_size = float(np.median(diffs)) if len(diffs) > 0 else 0.0
        p90_order_size = float(np.percentile(diffs, 90)) if len(diffs) > 0 else 0.0
        annualized_turnover = mean_order_size * 252.0
        return {
            "mean_abs_exposure": mean_abs_exposure,
            "mean_order_size": mean_order_size,
            "median_order_size": median_order_size,
            "p90_order_size": p90_order_size,
            "annualized_turnover": annualized_turnover,
        }

    expo_train_series = pd.Series(expo_train, index=idx_train)
    expo_test_series = pd.Series(expo_test, index=idx_test)
    train_expo_stats = exposure_stats(expo_train_series)
    test_expo_stats = exposure_stats(expo_test_series)

    # scaling indicator (test)
    current_meta_vol = float(ml_meta_test_metrics.get("annualized_vol", 0.0))
    baseline_ml_vol = float(ml_test_metrics.get("annualized_vol", 0.0))
    vol_scale_to_match_ml = baseline_ml_vol / current_meta_vol if current_meta_vol > 0 else float("inf")
    vol_scale_suggestion = min(vol_scale_to_match_ml, 3.0) if current_meta_vol > 0 else 0.0

    # assemble execution stats and save
    exec_rows = []
    exec_rows.append({"split": "train", "ml_meta_annualized_vol": float(ml_meta_train_metrics.get("annualized_vol", 0.0)), "ml_annualized_vol": float(train_ml_metrics.get("annualized_vol", 0.0)), **train_expo_stats})
    exec_rows.append({"split": "test", "ml_meta_annualized_vol": float(ml_meta_test_metrics.get("annualized_vol", 0.0)), "ml_annualized_vol": float(ml_test_metrics.get("annualized_vol", 0.0)), **test_expo_stats})
    exec_df = pd.DataFrame(exec_rows).set_index("split")
    exec_df.to_csv(os.path.join(base, "ml_meta_execution_stats.csv"))

    # print diagnostics and scaling suggestions
    print("Chosen method:", best.get("method"))
    print("Chosen params: k=%.3f f=%.3f" % (best.get("k", 0.0), best.get("f", 0.0)))
    print("Train utility: %.6f" % best.get("utility", 0.0))
    print(
        "Train sharpe: %.4f, train turnover/year: %.1f, train mean|exp|: %.3f, train ann_vol: %.4f"
        % (
            best.get("sharpe", 0.0),
            best.get("turnover_per_year", 0.0),
            best.get("mean_abs_exposure", 0.0),
            best.get("annualized_vol", 0.0),
        )
    )

    print("--- Exposure & Execution Stats ---")
    print("Train mean_abs_exposure: %.4f" % train_expo_stats["mean_abs_exposure"])
    print("Train mean_order_size: %.6f, median: %.6f, p90: %.6f, annualized_turnover: %.3f" % (train_expo_stats["mean_order_size"], train_expo_stats["median_order_size"], train_expo_stats["p90_order_size"], train_expo_stats["annualized_turnover"]))
    print("Test mean_abs_exposure: %.4f" % test_expo_stats["mean_abs_exposure"])
    print("Test mean_order_size: %.6f, median: %.6f, p90: %.6f, annualized_turnover: %.3f" % (test_expo_stats["mean_order_size"], test_expo_stats["median_order_size"], test_expo_stats["p90_order_size"], test_expo_stats["annualized_turnover"]))

    print("--- Vol scaling suggestion (test) ---")
    print("Current ML_META ann vol: %.6f" % current_meta_vol)
    print("Baseline ML ann vol: %.6f" % baseline_ml_vol)
    if current_meta_vol > 0:
        print("vol_scale_to_match_ml: %.4f" % vol_scale_to_match_ml)
        print("conservative capped suggestion (<=3.0): %.4f" % vol_scale_suggestion)
    else:
        print("Current ML_META vol is zero; cannot compute scaling suggestion.")

    print("Test Sharpe ML: %.4f" % ml_test_metrics["sharpe"])
    print("Test Sharpe ML_META: %.4f" % ml_meta_test_metrics["sharpe"])
    print("Test total return (ML): %.4f, vol: %.4f" % (ml_test_metrics["total_return"], ml_test_metrics["annualized_vol"]))
    print("Test total return (ML_META): %.4f, vol: %.4f" % (ml_meta_test_metrics["total_return"], ml_meta_test_metrics["annualized_vol"]))
    print("Delta (ML_META - ML) Sharpe on test: %.4f" % (ml_meta_test_metrics["sharpe"] - ml_test_metrics["sharpe"]))


if __name__ == "__main__":
    main()
