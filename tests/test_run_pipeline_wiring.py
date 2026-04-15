import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import run_pipeline


def test_run_modular_wires_new_knobs(monkeypatch):
    captured = {}

    def _fake_pipeline(run_id, cfg, out_root):
        captured["run_id"] = run_id
        captured["cfg"] = cfg
        captured["out_root"] = out_root
        return {"manifest_path": "x", "out_dir": "y", "manifest": {"run_id": run_id, "artifacts": []}}

    import tema.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _fake_pipeline)
    run_pipeline.run_modular(
        run_id="wired",
        out_root="outputs",
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        ml_modular_path_enabled=True,
        ml_probability_threshold=0.55,
        data_max_assets=11,
        data_full_universe_for_parity=False,
        portfolio_method="bl",
        portfolio_risk_aversion=3.0,
        portfolio_bl_tau=0.07,
        portfolio_bl_view_confidence=0.70,
        ml_hmm_scalar_floor=0.25,
        ml_hmm_scalar_ceiling=1.10,
        vol_target_apply_to_ml=True,
    )

    cfg = captured["cfg"]
    assert cfg.modular_data_signals_enabled is True
    assert cfg.portfolio_modular_enabled is True
    assert cfg.ml_modular_path_enabled is True
    assert cfg.ml_probability_threshold == 0.55
    assert cfg.data_max_assets == 11
    assert cfg.data_full_universe_for_parity is False
    assert cfg.portfolio_risk_aversion == 3.0
    assert cfg.portfolio_bl_tau == 0.07
    assert cfg.portfolio_bl_view_confidence == 0.70
    assert cfg.ml_hmm_scalar_floor == 0.25
    assert cfg.ml_hmm_scalar_ceiling == 1.10
    assert cfg.vol_target_apply_to_ml is True


def test_run_legacy_writes_performance_manifest(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    template_dir = fake_root / "Template"
    template_dir.mkdir(parents=True)
    (template_dir / "TEMA-TEMPLATE(NEW_).py").write_text("print('ok')", encoding="utf-8")

    def _fake_run_path(path, run_name=None, init_globals=None):
        csv_data = (
            "dataset,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown\n"
            "train,0,0.20,0.10,2.0,-0.10\n"
            "test,0,0.12,0.08,1.5,-0.12\n"
        )
        (template_dir / "bl_portfolio_metrics.csv").write_text(csv_data, encoding="utf-8")
        return {}

    monkeypatch.setattr(run_pipeline, "ROOT", fake_root)
    monkeypatch.setattr(run_pipeline.runpy, "run_path", _fake_run_path)
    monkeypatch.setenv("TEMA_RUN_LEGACY_EXECUTE", "1")

    out_root = tmp_path / "outputs"
    res = run_pipeline.run_legacy(run_id="legacy-test", out_root=str(out_root))

    perf_path = Path(res["out_dir"]) / "performance.json"
    manifest_path = Path(res["manifest_path"])
    assert perf_path.exists()
    assert manifest_path.exists()

    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert perf["sharpe"] == 1.5
    assert perf["annual_return"] == 0.12
    assert manifest["legacy_executed"] is True
    assert manifest["artifacts"] == ["performance"]
