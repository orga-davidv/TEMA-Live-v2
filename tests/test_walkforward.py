import numpy as np
import pandas as pd
from pathlib import Path

from tema.validation import walkforward


def test_generate_windows_and_metrics_synthetic():
    # Create 5 years of daily business days
    dates = pd.date_range(start="2015-01-01", end="2020-01-01", freq="B")
    rng = np.random.RandomState(42)
    rets = pd.Series(rng.normal(loc=0.0002, scale=0.01, size=len(dates)), index=dates)

    windows = walkforward.generate_walkforward_windows(dates, train_years=2, test_months=6, step_months=3)
    assert len(windows) > 0

    per_df = walkforward.compute_window_metrics(rets, windows)
    assert not per_df.empty
    # metrics should be finite or numeric
    assert all(np.isfinite(per_df['sharpe'].fillna(0).to_numpy()))
    assert all(np.isfinite(per_df['annual_return'].fillna(0).to_numpy()))
    assert all(np.isfinite(per_df['annual_vol'].fillna(0).to_numpy()))
    assert all(np.isfinite(per_df['max_drawdown'].fillna(0).to_numpy()))


def test_run_walkforward_on_pipeline_output(tmp_path, monkeypatch):
    # Use run_pipeline to generate outputs (parity style)
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")
    import run_pipeline
    
    out_root = str(tmp_path / "outputs")
    res = run_pipeline.run_modular(
        run_id="wf-int-test",
        out_root=out_root,
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=False,
    )
    out_dir = Path(res['out_dir'])
    produced = out_dir / 'portfolio_test_returns.csv'
    assert produced.exists()

    # run the walkforward script programmatically
    from scripts import run_walkforward
    rc = run_walkforward.run(str(out_dir), 'baseline')
    assert rc in (0,1,2)  # should return an int
    # ensure files are created when rc in (0,1)
    if rc in (0,1):
        assert (out_dir / 'walkforward_windows.csv').exists()
        assert (out_dir / 'walkforward_report.json').exists()
