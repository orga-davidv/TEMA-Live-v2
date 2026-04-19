#!/usr/bin/env python3
"""
Template/dynamic_scaler_compare.py
Implements three dynamic scaler variants (A/B/C), calibrates on train set and evaluates on test set.
Saves comparison CSVs and Plotly PNGs.
"""
import sys
from pathlib import Path

# Ensure working dir
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Template"

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# HMM
from hmmlearn.hmm import GaussianHMM

# Constants
TARGET_ANN_VOL = 0.10
TRADING_DAYS = 252
SCALAR_CAP = 50.0
LAG = 1  # lag indicators by 1 to avoid lookahead

# Files
train_ml_fp = TEMPLATE / "portfolio_train_returns_ml.csv"
test_ml_fp = TEMPLATE / "portfolio_test_returns_ml.csv"
train_fp = TEMPLATE / "portfolio_train_returns.csv"
test_fp = TEMPLATE / "portfolio_test_returns.csv"

out_compare = TEMPLATE / "dynamic_scaler_comparison.csv"
out_test_returns = TEMPLATE / "dynamic_scaler_test_returns.csv"
out_equity_png = TEMPLATE / "dynamic_scaler_equity_test.png"
out_dd_png = TEMPLATE / "dynamic_scaler_drawdown_test.png"

# Helper functions

def safe_read_csv(p):
    if p.exists():
        return pd.read_csv(p, parse_dates=True, index_col=0)
    return None


def compute_rsi(series, window=14):
    # series: price series
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def annualize_return(total_return, days):
    return (1 + total_return) ** (TRADING_DAYS / days) - 1


def annual_return_from_series(returns):
    # arithmetic approach using geometric mean
    daily_ret = returns.dropna()
    cum = (1 + daily_ret).prod()
    days = len(daily_ret)
    if days == 0:
        return np.nan
    return (cum ** (TRADING_DAYS / days)) - 1


def annual_vol(returns):
    return returns.std(ddof=0) * np.sqrt(TRADING_DAYS)


def sharpe(returns, rf=0.0):
    vol = annual_vol(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    ann_ret = annual_return_from_series(returns)
    return (ann_ret - rf) / vol


def max_drawdown(returns):
    # returns series
    eq = (1 + returns.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return dd.min()


def scalar_stats(scalar):
    s = scalar.dropna()
    if len(s) == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {"mean": s.mean(), "median": s.median(), "p90": np.percentile(s, 90), "max": s.max()}


def calibrate_scalar_on_train(base_returns_train, raw_scalar_train, cap=SCALAR_CAP, target_ann_vol=TARGET_ANN_VOL):
    # raw_scalar_train and base_returns_train indexed aligned
    scalar = raw_scalar_train.fillna(0).clip(lower=0)
    # Apply initial scalar to compute train vol
    scaled_returns = base_returns_train * scalar
    vol = annual_vol(scaled_returns)
    if vol <= 0 or np.isnan(vol):
        factor = 0.0
    else:
        factor = target_ann_vol / vol
    scalar_calibrated = (scalar * factor).clip(upper=cap)
    # Recompute final train vol
    final_vol = annual_vol(base_returns_train * scalar_calibrated)
    return scalar_calibrated, factor, vol, final_vol


def prepare_data():
    ml_train = safe_read_csv(train_ml_fp)
    ml_test = safe_read_csv(test_ml_fp)
    if ml_train is None or ml_test is None:
        print("Missing ML train/test CSVs in Template/. Please provide the required files.")
        sys.exit(2)
    # Assume returns column named 'return' or first column after index
    # If multiple columns, pick the first
    def pick_return_col(df):
        if df is None:
            return None
        if 'return' in df.columns:
            return df['return'].astype(float)
        else:
            return df.iloc[:,0].astype(float)
    base_train = pick_return_col(ml_train)
    base_test = pick_return_col(ml_test)
    # For context, optional BL series
    bl_train = safe_read_csv(train_fp)
    bl_test = safe_read_csv(test_fp)
    if bl_train is not None:
        bl_train = pick_return_col(bl_train)
    if bl_test is not None:
        bl_test = pick_return_col(bl_test)
    return base_train, base_test, bl_train, bl_test


def variant_A(base_train, base_test):
    all_series = pd.concat([base_train, base_test])
    all_series = all_series[~all_series.index.duplicated(keep='last')]
    # Momentum + RSI + rolling vol scaling
    # Build price series
    price_all = (1 + all_series).cumprod()
    # Momentum: price / price.shift(21) -1
    mom_w = 21
    mom_all = price_all / price_all.shift(mom_w) - 1
    # RSI on price with window 14
    rsi_all = compute_rsi(price_all, 14)
    # rolling vol
    vol_all = all_series.rolling(21).std(ddof=0)
    # Combine: positive momentum and RSI>50 produces signal
    combined_all = (mom_all.fillna(0)) * ((rsi_all - 50) / 50)
    # Lag indicators by 1
    combined_all = combined_all.shift(LAG)
    vol_all = vol_all.shift(LAG)
    # raw scalar = max(0, combined) / vol (to target higher scalar for low vol)
    raw_all = (combined_all.clip(lower=0)) / (vol_all + 1e-9)
    # Replace inf
    raw_all.replace([np.inf, -np.inf], 0, inplace=True)
    raw_train = raw_all.loc[base_train.index]
    raw_test = raw_all.loc[base_test.index]
    return raw_train, raw_test


def variant_B(base_train, base_test):
    all_series = pd.concat([base_train, base_test])
    all_series = all_series[~all_series.index.duplicated(keep='last')]
    # Regime scalar from rolling Sharpe
    roll = 63
    mean_all = all_series.rolling(roll).mean()
    std_all = all_series.rolling(roll).std(ddof=0)
    sharpe_all = (mean_all / (std_all + 1e-9)) * np.sqrt(TRADING_DAYS)
    # Lag
    sharpe_all = sharpe_all.shift(LAG)
    # Map sharpe to 0..1 via sigmoid-ish mapping
    clipped_all = np.tanh(sharpe_all / 2)  # between -1 and 1
    raw_all = (clipped_all + 1) / 2
    raw_train = raw_all.loc[base_train.index]
    raw_test = raw_all.loc[base_test.index]
    raw_train.fillna(0, inplace=True)
    raw_test.fillna(0, inplace=True)
    return raw_train, raw_test


def variant_C(base_train, base_test):
    # HMM state-probability scalar
    # Fit on train returns (dropna)
    rt = base_train.dropna().values.reshape(-1,1)
    if len(rt) < 10:
        # not enough data
        raw_train = pd.Series(0, index=base_train.index)
        raw_test = pd.Series(0, index=base_test.index)
        return raw_train, raw_test
    model = GaussianHMM(n_components=3, covariance_type='full', random_state=42, n_iter=200)
    model.fit(rt)
    # Determine which state is bull (higher mean)
    means = np.array([model.means_[i,0] for i in range(model.n_components)])
    bull_state = np.argmax(means)
    # Compute posteriors on continuous history to avoid test cold-start reset.
    # IMPORTANT: use forward-filtered probabilities (no lookahead), not smoothed posteriors.
    all_series = pd.concat([base_train, base_test])
    all_series = all_series[~all_series.index.duplicated(keep='last')]

    def forward_filter_probs_1d(model, X):
        """
        Forward-only state filtering for 1D Gaussian HMM:
        returns P(z_t | x_1..x_t) for each t (no future leakage).
        """
        n_states = model.n_components
        start = np.asarray(model.startprob_, dtype=float)
        trans = np.asarray(model.transmat_, dtype=float)
        means = np.asarray(model.means_[:, 0], dtype=float)

        # Support common hmmlearn covariance layouts for 1D.
        cov = np.asarray(model.covars_, dtype=float)
        if cov.ndim == 3:  # full covariance [state, dim, dim]
            vars_ = cov[:, 0, 0]
        elif cov.ndim == 2:  # diag covariance [state, dim]
            vars_ = cov[:, 0]
        else:  # spherical or already collapsed
            vars_ = cov.reshape(-1)

        vars_ = np.maximum(vars_, 1e-12)
        x = X.reshape(-1, 1)
        # Emission likelihoods b_t(i)
        norm = np.sqrt(2.0 * np.pi * vars_)
        emis = np.exp(-0.5 * ((x - means) ** 2) / vars_) / norm
        emis = np.maximum(emis, 1e-300)

        probs = np.zeros((x.shape[0], n_states), dtype=float)
        alpha = start * emis[0]
        alpha_sum = alpha.sum()
        if alpha_sum <= 0 or not np.isfinite(alpha_sum):
            alpha = np.full(n_states, 1.0 / n_states, dtype=float)
        else:
            alpha = alpha / alpha_sum
        probs[0] = alpha

        for t in range(1, x.shape[0]):
            alpha = (alpha @ trans) * emis[t]
            s = alpha.sum()
            if s <= 0 or not np.isfinite(s):
                alpha = np.full(n_states, 1.0 / n_states, dtype=float)
            else:
                alpha = alpha / s
            probs[t] = alpha
        return probs

    def posterior_probs(model, series):
        idx = series.dropna().index
        X = series.dropna().values.reshape(-1,1)
        probs = forward_filter_probs_1d(model, X)
        pbull = probs[:, bull_state]
        s = pd.Series(index=idx, data=pbull)
        return s
    post_all = posterior_probs(model, all_series)
    # Lag by 1
    post_all = post_all.shift(LAG)
    post_train = post_all.loc[base_train.index]
    post_test = post_all.loc[base_test.index]
    post_train = post_train.reindex(base_train.index).fillna(0)
    post_test = post_test.reindex(base_test.index).fillna(0)
    raw_train = post_train
    raw_test = post_test
    return raw_train, raw_test


def evaluate_variant(name, base_train, base_test, raw_train, raw_test):
    # Calibrate on train to reach target vol
    scalar_train_cal, factor, pre_vol, post_vol = calibrate_scalar_on_train(base_train, raw_train)
    # Apply same factor (but cap) to test raw
    scalar_test = (raw_test.fillna(0) * factor).clip(upper=SCALAR_CAP)
    # Compute scaled returns
    scaled_train = base_train * scalar_train_cal
    scaled_test = base_test * scalar_test
    # Metrics
    def metrics(returns_series):
        total_ret = (1 + returns_series.dropna()).prod() - 1 if len(returns_series.dropna())>0 else np.nan
        ann_ret = annual_return_from_series(returns_series)
        ann_vol = annual_vol(returns_series)
        s = sharpe(returns_series)
        mdd = max_drawdown(returns_series)
        return dict(total_return=total_ret, annual_return=ann_ret, annual_vol=ann_vol, sharpe=s, max_drawdown=mdd)
    metrics_train = metrics(scaled_train)
    metrics_test = metrics(scaled_test)
    stats_train = scalar_stats(scalar_train_cal)
    stats_test = scalar_stats(scalar_test)
    out = {
        'name': name,
        'factor': factor,
        'pre_vol': pre_vol,
        'post_vol': post_vol,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'scalar_stats_train': stats_train,
        'scalar_stats_test': stats_test,
        'series_test_returns': scaled_test,
        'series_train_returns': scaled_train,
    }
    return out


def make_plots(df_returns, out_equity, out_dd):
    # df_returns: DataFrame columns = variants, index = dates
    fig = go.Figure()
    for c in df_returns.columns:
        eq = (1 + df_returns[c].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=c))
    fig.update_layout(title='Equity curves (test)', xaxis_title='Date', yaxis_title='Cumulative Return')
    fig.write_image(str(out_equity))
    # Drawdowns
    fig2 = go.Figure()
    for c in df_returns.columns:
        eq = (1 + df_returns[c].fillna(0)).cumprod()
        peak = pd.Series(eq).cummax().values
        dd = (eq.values - peak) / peak
        fig2.add_trace(go.Scatter(x=eq.index, y=dd, name=c))
    fig2.update_layout(title='Drawdowns (test)', xaxis_title='Date', yaxis_title='Drawdown')
    fig2.write_image(str(out_dd))


def main():
    base_train, base_test, bl_train, bl_test = prepare_data()
    # Ensure indices are datetime
    base_train.index = pd.to_datetime(base_train.index)
    base_test.index = pd.to_datetime(base_test.index)

    # Build variants
    rawA_train, rawA_test = variant_A(base_train, base_test)
    rawB_train, rawB_test = variant_B(base_train, base_test)
    rawC_train, rawC_test = variant_C(base_train, base_test)

    resA = evaluate_variant('A_mom_rsi_vol', base_train, base_test, rawA_train, rawA_test)
    resB = evaluate_variant('B_rolling_sharpe', base_train, base_test, rawB_train, rawB_test)
    resC = evaluate_variant('C_hmm_prob', base_train, base_test, rawC_train, rawC_test)

    results = [resA, resB, resC]

    # Save comparison CSV
    rows = []
    for r in results:
        row = {
            'variant': r['name'],
            'factor': r['factor'],
            'pre_vol': r['pre_vol'],
            'post_vol': r['post_vol'],
            'train_total_return': r['metrics_train']['total_return'],
            'train_annual_return': r['metrics_train']['annual_return'],
            'train_annual_vol': r['metrics_train']['annual_vol'],
            'train_sharpe': r['metrics_train']['sharpe'],
            'train_max_drawdown': r['metrics_train']['max_drawdown'],
            'test_total_return': r['metrics_test']['total_return'],
            'test_annual_return': r['metrics_test']['annual_return'],
            'test_annual_vol': r['metrics_test']['annual_vol'],
            'test_sharpe': r['metrics_test']['sharpe'],
            'test_max_drawdown': r['metrics_test']['max_drawdown'],
        }
        row.update({f'train_scalar_{k}': v for k, v in r['scalar_stats_train'].items()})
        row.update({f'test_scalar_{k}': v for k, v in r['scalar_stats_test'].items()})
        rows.append(row)
    dfcomp = pd.DataFrame(rows)
    dfcomp.to_csv(out_compare, index=False)

    # Save test returns per variant
    df_returns = pd.DataFrame({r['name']: r['series_test_returns'] for r in results})
    df_returns.to_csv(out_test_returns)

    # Plots
    make_plots(df_returns, out_equity_png, out_dd_png)

    # Print concise ranking by test sharpe and test annual return
    ranking_sharpe = dfcomp.sort_values('test_sharpe', ascending=False)[['variant','test_sharpe']]
    ranking_return = dfcomp.sort_values('test_annual_return', ascending=False)[['variant','test_annual_return']]
    print('\nRanking by test Sharpe:')
    print(ranking_sharpe.to_string(index=False))
    print('\nRanking by test Annual Return:')
    print(ranking_return.to_string(index=False))

    # Also print top results concisely
    best_sharpe = ranking_sharpe.iloc[0]['variant']
    best_return = ranking_return.iloc[0]['variant']
    print(f"\nBest by Sharpe: {best_sharpe}")
    print(f"Best by Return: {best_return}")

if __name__ == '__main__':
    main()
