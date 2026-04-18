import json
from pathlib import Path


def test_dd_guard_enabled_does_not_crash_and_writes_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")

    from tema.config import BacktestConfig
    from tema.pipeline import run_pipeline

    out_root = str(tmp_path / "outputs")
    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=False,
        dd_guard_enabled=True,
        # Force a breach so the overlay is applied.
        dd_guard_max_drawdown=0.0,
        dd_guard_floor=0.5,
        dd_guard_recovery_halflife=1,
    )

    res = run_pipeline(run_id="dd-guard-int-test", cfg=cfg, out_root=out_root)
    out_dir = Path(res["out_dir"])

    dd_path = out_dir / "dd_guard.json"
    assert dd_path.exists(), "dd_guard.json was not written"

    info = json.loads(dd_path.read_text(encoding="utf-8"))
    assert info.get("enabled") is True
    assert info.get("allow_full_derisk") is True
    assert info.get("applied") is True
    assert int(info.get("test_periods", 0)) > 0
