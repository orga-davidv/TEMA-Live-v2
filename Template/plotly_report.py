#!/usr/bin/env python3
"""
plotly_report.py

Loads portfolio test returns CSVs from this Template/ directory and writes PNG charts using Plotly + kaleido:
 - plotly_equity_curves_test.png
 - plotly_drawdown_curves_test.png
 - plotly_returns_distribution_test.png

Usage: python3 Template/plotly_report.py
If packages are missing, install them with pip (the wrapper runner will do that).
"""

import os
import glob
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    import plotly.io as pio
except Exception as e:
    print("Missing libraries when importing:", e)
    print("Please install pandas, plotly and kaleido (e.g. pip install pandas plotly kaleido)")
    raise

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = [
    (BASE_DIR / "portfolio_test_returns.csv", "BL"),
    (BASE_DIR / "portfolio_test_returns_ml.csv", "ML"),
    (BASE_DIR / "portfolio_test_returns_ml_meta.csv", "ML_META"),
]

OUT_EQ = BASE_DIR / "plotly_equity_curves_test.png"
OUT_DD = BASE_DIR / "plotly_drawdown_curves_test.png"
OUT_DIST = BASE_DIR / "plotly_returns_distribution_test.png"


def load_returns(path):
    """Load CSV into DataFrame. If a single column, keep name; if index is date-like, parse it."""
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        # fallback: read without index
        df = pd.read_csv(path)
    # If single unnamed column, try to rename
    if df.columns.tolist() == ["Unnamed: 0"] and df.shape[1] == 1:
        df = df.rename(columns={"Unnamed: 0": path.stem})
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def combine_sources(files):
    parts = []
    for p, tag in files:
        if p.exists():
            try:
                df = load_returns(p)
                # Prefix columns with tag to avoid collisions when merging
                df = df.copy()
                df.columns = [f"{tag}:{c}" for c in df.columns]
                parts.append(df)
                print(f"Loaded {p} -> columns: {list(df.columns)}")
            except Exception as e:
                print(f"Failed to load {p}: {e}")
        else:
            print(f"File not found (skipping): {p}")
    if not parts:
        return pd.DataFrame()
    # outer join on index
    combined = pd.concat(parts, axis=1).sort_index()
    return combined


def make_equity_curve(returns_df):
    # treat returns as simple returns (not log). Fill forward/backward missing values carefully
    returns_df = returns_df.fillna(0)
    equity = (1 + returns_df).cumprod()
    return equity


def make_drawdown(equity_df):
    running_max = equity_df.cummax()
    drawdown = equity_df.div(running_max) - 1
    return drawdown


def plot_equity(equity_df, outpath):
    if equity_df.empty:
        print("No equity data to plot for equity curves.")
        return False
    fig = go.Figure()
    for col in equity_df.columns:
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df[col], mode="lines", name=col))
    fig.update_layout(title="Equity Curves (Test)", xaxis_title="Date", yaxis_title="Equity (normalized)")
    try:
        pio.write_image(fig, str(outpath), format='png', engine='kaleido')
        print(f"Wrote equity chart: {outpath}")
        return True
    except Exception as e:
        print(f"Failed to write equity chart: {e}")
        return False


def plot_drawdown(drawdown_df, outpath):
    if drawdown_df.empty:
        print("No drawdown data to plot.")
        return False
    fig = go.Figure()
    for col in drawdown_df.columns:
        fig.add_trace(go.Scatter(x=drawdown_df.index, y=drawdown_df[col], mode="lines", name=col))
    fig.update_layout(title="Drawdown Curves (Test)", xaxis_title="Date", yaxis_title="Drawdown")
    try:
        pio.write_image(fig, str(outpath), format='png', engine='kaleido')
        print(f"Wrote drawdown chart: {outpath}")
        return True
    except Exception as e:
        print(f"Failed to write drawdown chart: {e}")
        return False


def plot_returns_distribution(returns_df, outpath):
    if returns_df.empty:
        print("No returns data to plot.")
        return False
    fig = go.Figure()
    # Overlay histograms for each column
    for col in returns_df.columns:
        fig.add_trace(go.Histogram(x=returns_df[col].dropna(), name=col, opacity=0.6, nbinsx=50))
    fig.update_layout(barmode='overlay', title='Returns Distribution (Test)', xaxis_title='Return', yaxis_title='Count')
    try:
        pio.write_image(fig, str(outpath), format='png', engine='kaleido')
        print(f"Wrote returns distribution chart: {outpath}")
        return True
    except Exception as e:
        print(f"Failed to write returns distribution chart: {e}")
        return False


def main():
    combined = combine_sources(INPUT_FILES)
    if combined.empty:
        print("No input returns files found. Exiting without creating plots.")
        sys.exit(2)

    equity = make_equity_curve(combined)
    dd = make_drawdown(equity)

    ok1 = plot_equity(equity, OUT_EQ)
    ok2 = plot_drawdown(dd, OUT_DD)
    ok3 = plot_returns_distribution(combined, OUT_DIST)

    success = all([ok1, ok2, ok3])
    if not success:
        print("One or more plots failed to generate. See messages above.")
        # exit with non-zero so caller can detect
        sys.exit(3)
    print("All plots generated successfully.")


if __name__ == '__main__':
    main()
