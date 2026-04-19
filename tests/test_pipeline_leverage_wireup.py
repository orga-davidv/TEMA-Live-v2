import json
from pathlib import Path

import numpy as np

from tema.config import BacktestConfig
from tema.pipeline import run_pipeline
from tema.pipeline import runner as pipeline_runner


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _build_small_panel(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        data_dir / "a_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,100\n"
        "1677715200000,101\n"
        "1677801600000,102\n"
        "1678060800000,103\n"
        "1678147200000,104\n"
        "1678233600000,105\n",
    )
    _write_csv(
        data_dir / "b_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,60\n"
        "1677715200000,59\n"
        "1677801600000,58\n"
        "1678060800000,57\n"
        "1678147200000,56\n"
        "1678233600000,55\n",
    )


def _base_cfg(data_dir: Path) -> BacktestConfig:
    return BacktestConfig(
        modular_data_signals_enabled=True,
        data_path=str(data_dir),
        data_max_assets=2,
        data_min_rows=4,
        data_train_ratio=0.8,
        signal_fast_period=2,
        signal_slow_period=3,
        signal_method="ema",
        ml_enabled=True,
        ml_modular_path_enabled=True,
        ml_probability_threshold=0.55,
        vol_target_enabled=False,
    )


def test_pipeline_leverage_stage_toggle_off_on(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg_off = _base_cfg(data_dir)
    cfg_off.cle_enabled = False
    res_off = run_pipeline(run_id="cle-toggle-off", cfg=cfg_off, out_root=str(out_root))
    info_off = json.loads((Path(res_off["out_dir"]) / "leverage_info.json").read_text(encoding="utf-8"))
    final_off = json.loads((Path(res_off["out_dir"]) / "final_weights.json").read_text(encoding="utf-8"))
    leveraged_off = json.loads((Path(res_off["out_dir"]) / "leveraged_final_weights.json").read_text(encoding="utf-8"))
    assert info_off["enabled"] is False
    assert np.allclose(final_off, leveraged_off, atol=1e-12)

    cfg_on = _base_cfg(data_dir)
    cfg_on.cle_enabled = True
    cfg_on.cle_base_leverage = 1.8
    cfg_on.cle_mapping_min_multiplier = 1.0
    cfg_on.cle_mapping_max_multiplier = 1.0
    res_on = run_pipeline(run_id="cle-toggle-on", cfg=cfg_on, out_root=str(out_root))
    info_on = json.loads((Path(res_on["out_dir"]) / "leverage_info.json").read_text(encoding="utf-8"))
    final_on = json.loads((Path(res_on["out_dir"]) / "final_weights.json").read_text(encoding="utf-8"))
    leveraged_on = json.loads((Path(res_on["out_dir"]) / "leveraged_final_weights.json").read_text(encoding="utf-8"))

    assert info_on["enabled"] is True
    assert info_on["applied"] is True
    assert abs(info_on["leverage_scalar"] - 1.8) < 1e-12
    assert not np.allclose(final_on, leveraged_on, atol=1e-12)


def test_pipeline_leverage_cap_clipping_and_event_gating(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg = _base_cfg(data_dir)
    cfg.cle_enabled = True
    cfg.cle_base_leverage = 5.0
    cfg.cle_mapping_min_multiplier = 1.0
    cfg.cle_mapping_max_multiplier = 1.0
    cfg.cle_force_event_blackout = True
    cfg.cle_gate_event_blackout_cap = 0.4
    cfg.cle_leverage_cap = 0.3
    cfg.cle_leverage_floor = 0.0

    res = run_pipeline(run_id="cle-gated", cfg=cfg, out_root=str(out_root))
    info = json.loads((Path(res["out_dir"]) / "leverage_info.json").read_text(encoding="utf-8"))

    assert info["enabled"] is True
    assert info["gate_context"]["event_blackout"] is True
    assert info["engine_diagnostics"]["gate_flags"]["event_blackout"] is True
    assert abs(info["leverage_scalar"] - 0.3) < 1e-12
    assert info["leveraged_gross_exposure"] < info["base_gross_exposure"]


def test_pipeline_leverage_floor_invariant_is_enforced(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg = _base_cfg(data_dir)
    cfg.cle_enabled = True
    cfg.cle_base_leverage = 0.0
    cfg.cle_mapping_min_multiplier = 0.0
    cfg.cle_mapping_max_multiplier = 0.0
    cfg.cle_leverage_floor = 0.75
    cfg.cle_leverage_cap = 1.25

    res = run_pipeline(run_id="cle-floor", cfg=cfg, out_root=str(out_root))
    info = json.loads((Path(res["out_dir"]) / "leverage_info.json").read_text(encoding="utf-8"))

    assert info["enabled"] is True
    assert abs(info["leverage_scalar"] - 0.75) < 1e-12
    assert info["leverage_scalar"] >= cfg.cle_leverage_floor
    assert info["leverage_scalar"] <= cfg.cle_leverage_cap


def test_pipeline_cle_fallback_keeps_parity_with_base_weights(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg = _base_cfg(data_dir)
    cfg.cle_enabled = True
    cfg.cle_mapping_mode = "invalid-mode"
    res = run_pipeline(run_id="cle-fallback-parity", cfg=cfg, out_root=str(out_root))
    out_dir = Path(res["out_dir"])

    info = json.loads((out_dir / "leverage_info.json").read_text(encoding="utf-8"))
    final_weights = json.loads((out_dir / "final_weights.json").read_text(encoding="utf-8"))
    leveraged_weights = json.loads((out_dir / "leveraged_final_weights.json").read_text(encoding="utf-8"))

    assert info["enabled"] is True
    assert info["applied"] is False
    assert info["reason"] == "fallback_to_base_weights"
    assert "error" in info and info["error"]
    assert np.allclose(final_weights, leveraged_weights, atol=1e-12)


def test_pipeline_writes_cle_report_artifact_with_explainability_fields(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg = _base_cfg(data_dir)
    cfg.cle_enabled = True
    cfg.cle_base_leverage = 1.9
    cfg.cle_force_event_blackout = True
    cfg.cle_gate_event_blackout_cap = 0.4
    cfg.cle_mapping_mode = "linear"
    cfg.cle_mapping_min_multiplier = 0.9
    cfg.cle_mapping_max_multiplier = 1.3

    res = run_pipeline(run_id="cle-reporting", cfg=cfg, out_root=str(out_root))
    out_dir = Path(res["out_dir"])

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "cle_report" in manifest.get("artifacts", [])

    report = json.loads((out_dir / "cle_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "cle_report.v1"
    assert isinstance(report["C_t"], float)
    assert isinstance(report["m_t"], float)
    assert isinstance(report["component_contributions"], list)
    assert len(report["component_contributions"]) > 0
    first_component = report["component_contributions"][0]
    assert {
        "name",
        "raw_signal",
        "aligned_signal",
        "normalized_signal",
        "weight",
        "contribution",
    }.issubset(first_component.keys())
    assert isinstance(report["triggered_gates"], list)
    assert any(g["gate"] == "event_blackout" for g in report["triggered_gates"])
    assert all(isinstance(g.get("reason"), str) and g["reason"] for g in report["triggered_gates"])


def test_leverage_stage_is_deterministic_for_fixed_inputs():
    cfg = BacktestConfig(
        cle_enabled=True,
        cle_base_leverage=1.25,
        cle_mapping_mode="linear",
        cle_mapping_min_multiplier=0.9,
        cle_mapping_max_multiplier=1.2,
    )
    base_weights = [0.3, 0.4, 0.3]
    expected_alphas = [0.02, -0.01, 0.015]
    ml_info = {
        "scalar": [1.1, 0.95, 1.05],
        "decisions": [1.0, 1.0, 0.0],
        "vol_target": {"applied": True, "leverage": 0.9},
    }
    ensemble_info = {
        "weights": {"tema_base": 0.5, "ml_proxy": 0.3, "risk_proxy": 0.2},
        "regime_score": 0.35,
    }

    weights_1, info_1 = pipeline_runner._leverage_stage(
        cfg=cfg,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.92,
    )
    weights_2, info_2 = pipeline_runner._leverage_stage(
        cfg=cfg,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.92,
    )

    assert np.allclose(weights_1, weights_2, atol=1e-15)
    assert abs(info_1["leverage_scalar"] - info_2["leverage_scalar"]) < 1e-15
    assert np.allclose(info_1["per_asset_leverage"], info_2["per_asset_leverage"], atol=1e-15)


def test_leverage_stage_online_calibration_flag_off_on():
    base_weights = [0.35, 0.30, 0.20, 0.15]
    expected_alphas = [0.02, -0.03, 0.01, 0.025]
    ml_info = {
        "scalar": [1.1, 0.9, 1.05, 0.95],
        "decisions": [1.0, 0.0, 1.0, 1.0],
        "vol_target": {"applied": True, "leverage": 0.95},
    }
    ensemble_info = {
        "weights": {"tema_base": 0.55, "ml_proxy": 0.25, "risk_proxy": 0.20},
        "regime_score": 0.2,
    }

    cfg_off = BacktestConfig(cle_enabled=True, cle_online_calibration_enabled=False, cle_policy_seed=17)
    _, info_off = pipeline_runner._leverage_stage(
        cfg=cfg_off,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.9,
    )
    assert info_off["online_calibration"]["enabled"] is False
    assert info_off["online_calibration"]["applied"] is False

    cfg_on = BacktestConfig(
        cle_enabled=True,
        cle_online_calibration_enabled=True,
        cle_online_calibration_window=2,
        cle_policy_seed=17,
    )
    _, info_on = pipeline_runner._leverage_stage(
        cfg=cfg_on,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.9,
    )
    assert info_on["online_calibration"]["enabled"] is True
    assert info_on["online_calibration"]["applied"] is True
    assert info_on["online_calibration"]["n_updates"] > 0
    assert info_on["online_calibration"]["learned_global_coefficients"] != info_off["online_calibration"]["learned_global_coefficients"]


def test_leverage_stage_online_calibration_is_seed_deterministic():
    cfg = BacktestConfig(
        cle_enabled=True,
        cle_online_calibration_enabled=True,
        cle_online_calibration_window=2,
        cle_policy_seed=11,
    )
    base_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    expected_alphas = [0.03, -0.01, 0.02, -0.015, 0.01]
    ml_info = {
        "scalar": [1.1, 0.95, 1.0, 1.05, 0.9],
        "decisions": [1.0, 1.0, 0.0, 1.0, 0.0],
        "vol_target": {"applied": True, "leverage": 0.9},
    }
    ensemble_info = {"weights": {"tema_base": 0.5, "ml_proxy": 0.35, "risk_proxy": 0.15}, "regime_score": 0.1}

    weights_1, info_1 = pipeline_runner._leverage_stage(
        cfg=cfg,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.92,
    )
    weights_2, info_2 = pipeline_runner._leverage_stage(
        cfg=cfg,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.92,
    )
    assert np.allclose(weights_1, weights_2, atol=1e-15)
    assert info_1["online_calibration"]["learner_state"] == info_2["online_calibration"]["learner_state"]

    cfg_other_seed = BacktestConfig(
        cle_enabled=True,
        cle_online_calibration_enabled=True,
        cle_online_calibration_window=2,
        cle_policy_seed=99,
    )
    _, info_other = pipeline_runner._leverage_stage(
        cfg=cfg_other_seed,
        base_weights=base_weights,
        expected_alphas=expected_alphas,
        ml_info=ml_info,
        ensemble_info=ensemble_info,
        dd_guard_last_scalar=0.92,
    )
    assert info_1["online_calibration"]["learner_state"]["weights"] != info_other["online_calibration"]["learner_state"]["weights"]


def test_pipeline_persists_online_calibration_artifact_when_enabled(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg = _base_cfg(data_dir)
    cfg.cle_enabled = True
    cfg.cle_policy_seed = 19
    cfg.cle_online_calibration_enabled = True
    cfg.cle_online_calibration_window = 1
    res = run_pipeline(run_id="cle-online-calibration", cfg=cfg, out_root=str(out_root))
    out_dir = Path(res["out_dir"])

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "cle_online_calibration" in manifest["artifacts"]

    payload = json.loads((out_dir / "cle_online_calibration.json").read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["seed"] == 19
    assert payload["n_updates"] > 0


def test_pipeline_online_calibration_is_deterministic_for_fixed_seed(tmp_path):
    data_dir = tmp_path / "merged_d1"
    _build_small_panel(data_dir)
    out_root = tmp_path / "outputs"

    cfg = _base_cfg(data_dir)
    cfg.cle_enabled = True
    cfg.cle_policy_seed = 23
    cfg.cle_online_calibration_enabled = True
    cfg.cle_online_calibration_window = 2

    run_a = run_pipeline(run_id="cle-seed-a", cfg=cfg, out_root=str(out_root))
    run_b = run_pipeline(run_id="cle-seed-b", cfg=cfg, out_root=str(out_root))
    out_a = Path(run_a["out_dir"])
    out_b = Path(run_b["out_dir"])

    leveraged_a = json.loads((out_a / "leveraged_final_weights.json").read_text(encoding="utf-8"))
    leveraged_b = json.loads((out_b / "leveraged_final_weights.json").read_text(encoding="utf-8"))
    info_a = json.loads((out_a / "leverage_info.json").read_text(encoding="utf-8"))
    info_b = json.loads((out_b / "leverage_info.json").read_text(encoding="utf-8"))

    assert np.allclose(leveraged_a, leveraged_b, atol=1e-15)
    assert abs(info_a["leverage_scalar"] - info_b["leverage_scalar"]) < 1e-15
    assert info_a["online_calibration"]["learner_state"] == info_b["online_calibration"]["learner_state"]
