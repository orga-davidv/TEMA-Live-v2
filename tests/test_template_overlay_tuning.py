from dataclasses import replace

import numpy as np
import pandas as pd

from tema.config import BacktestConfig
import tema.ml.template_overlay as overlay_mod


def _passthrough_scalar(*, cfg, ml_train_rets, ml_test_rets, **kwargs):
    return ml_train_rets, ml_test_rets, {"enabled": False}


def _passthrough_vol_target(*, cfg, train_port_rets, test_port_rets, ml_train_rets, ml_test_rets):
    return train_port_rets, test_port_rets, ml_train_rets, ml_test_rets, {"enabled": False}


def test_select_computed_overlay_cfg_picks_best_candidate(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    train_port = pd.Series(np.linspace(-0.01, 0.015, len(idx)), index=idx)

    def _fake_apply_hmm_softprob_rf_strategy(*, train_port_rets, test_port_rets, cfg, hmm_engine):
        base = 0.0001
        alpha = (
            base
            + 0.00003 * float(cfg.rf_max_depth)
            + 0.00002 * float(cfg.hmm_n_states)
            + 0.00001 * float(cfg.ml_target_exposure)
        )
        wobble_train = 0.0001 * np.sin(np.linspace(0.0, 6.0, len(train_port_rets)))
        wobble_test = 0.0001 * np.sin(np.linspace(0.0, 6.0, len(test_port_rets)))
        ml_train = pd.Series(alpha + wobble_train, index=train_port_rets.index)
        ml_test = pd.Series(alpha + wobble_test, index=test_port_rets.index)
        diag = {"test_avg_exposure": float(cfg.ml_target_exposure)}
        return ml_train, ml_test, diag, pd.DataFrame(), {}

    monkeypatch.setattr(overlay_mod, "apply_hmm_softprob_rf_strategy", _fake_apply_hmm_softprob_rf_strategy)

    cfg = BacktestConfig(
        template_use_precomputed_artifacts=False,
        ml_computed_overlay_tuning_enabled=True,
        ml_computed_overlay_tuning_min_rows=60,
        ml_computed_overlay_grid_rf_n_estimators=(300,),
        ml_computed_overlay_grid_rf_max_depth=(4, 6),
        ml_computed_overlay_grid_rf_min_samples_leaf=(20,),
        ml_computed_overlay_grid_target_exposure=(0.10, 0.40),
        ml_computed_overlay_grid_hmm_n_states=(2, 3),
    )

    selected_cfg, diag = overlay_mod.select_computed_overlay_cfg(
        train_port_rets=train_port,
        cfg=cfg,
        hmm_engine=object(),
    )

    assert diag["applied"] is True
    assert diag["validation_folds"] >= 1
    assert selected_cfg.rf_max_depth == 6
    assert selected_cfg.hmm_n_states == 3
    assert selected_cfg.ml_target_exposure == 0.40


def test_select_computed_overlay_cfg_penalizes_overfit_gap(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    train_port = pd.Series(np.linspace(-0.01, 0.015, len(idx)), index=idx)

    def _fake_apply_hmm_softprob_rf_strategy(*, train_port_rets, test_port_rets, cfg, hmm_engine):
        wobble_train = 0.0002 * np.sin(np.linspace(0.0, 8.0, len(train_port_rets)))
        wobble_test = 0.0002 * np.sin(np.linspace(0.0, 8.0, len(test_port_rets)))
        if int(cfg.rf_max_depth) == 6:
            ml_train = pd.Series(0.0040 + wobble_train, index=train_port_rets.index)
            ml_test = pd.Series(-0.0005 + wobble_test, index=test_port_rets.index)
        else:
            ml_train = pd.Series(0.0012 + wobble_train, index=train_port_rets.index)
            ml_test = pd.Series(0.0010 + wobble_test, index=test_port_rets.index)
        diag = {"test_avg_exposure": float(cfg.ml_target_exposure)}
        return ml_train, ml_test, diag, pd.DataFrame(), {}

    monkeypatch.setattr(overlay_mod, "apply_hmm_softprob_rf_strategy", _fake_apply_hmm_softprob_rf_strategy)

    cfg = BacktestConfig(
        template_use_precomputed_artifacts=False,
        ml_computed_overlay_tuning_enabled=True,
        ml_computed_overlay_tuning_min_rows=60,
        ml_computed_overlay_tuning_folds=3,
        ml_computed_overlay_tuning_overfit_penalty=0.75,
        ml_computed_overlay_grid_rf_n_estimators=(300,),
        ml_computed_overlay_grid_rf_max_depth=(4, 6),
        ml_computed_overlay_grid_rf_min_samples_leaf=(20,),
        ml_computed_overlay_grid_target_exposure=(0.20,),
        ml_computed_overlay_grid_hmm_n_states=(2,),
    )

    selected_cfg, diag = overlay_mod.select_computed_overlay_cfg(
        train_port_rets=train_port,
        cfg=cfg,
        hmm_engine=object(),
    )

    assert diag["applied"] is True
    assert selected_cfg.rf_max_depth == 4


def test_select_computed_overlay_cfg_overfit_penalty_changes_winner(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    train_port = pd.Series(np.linspace(-0.01, 0.015, len(idx)), index=idx)

    def _fake_apply_hmm_softprob_rf_strategy(*, train_port_rets, test_port_rets, cfg, hmm_engine):
        if int(cfg.rf_max_depth) == 6:
            # Better validation Sharpe but much larger train/val gap.
            ml_train = pd.Series(np.full(len(train_port_rets), 0.0030), index=train_port_rets.index)
            ml_test = pd.Series(np.full(len(test_port_rets), 0.0020), index=test_port_rets.index)
        else:
            # Slightly worse validation Sharpe, but near-zero overfit gap.
            ml_train = pd.Series(np.full(len(train_port_rets), 0.0020), index=train_port_rets.index)
            ml_test = pd.Series(np.full(len(test_port_rets), 0.0019), index=test_port_rets.index)
        diag = {"test_avg_exposure": float(cfg.ml_target_exposure)}
        return ml_train, ml_test, diag, pd.DataFrame(), {}

    monkeypatch.setattr(overlay_mod, "apply_hmm_softprob_rf_strategy", _fake_apply_hmm_softprob_rf_strategy)
    monkeypatch.setattr(
        overlay_mod,
        "_overlay_tuning_sharpe",
        lambda series, **kwargs: float(series.mean()) if len(series) else float("-inf"),
    )

    base_cfg = BacktestConfig(
        template_use_precomputed_artifacts=False,
        ml_computed_overlay_tuning_enabled=True,
        ml_computed_overlay_tuning_min_rows=60,
        ml_computed_overlay_tuning_folds=3,
        ml_computed_overlay_grid_rf_n_estimators=(300,),
        ml_computed_overlay_grid_rf_max_depth=(4, 6),
        ml_computed_overlay_grid_rf_min_samples_leaf=(20,),
        ml_computed_overlay_grid_target_exposure=(0.20,),
        ml_computed_overlay_grid_hmm_n_states=(2,),
    )

    selected_no_penalty, _ = overlay_mod.select_computed_overlay_cfg(
        train_port_rets=train_port,
        cfg=replace(base_cfg, ml_computed_overlay_tuning_overfit_penalty=0.0),
        hmm_engine=object(),
    )
    selected_with_penalty, _ = overlay_mod.select_computed_overlay_cfg(
        train_port_rets=train_port,
        cfg=replace(base_cfg, ml_computed_overlay_tuning_overfit_penalty=0.5),
        hmm_engine=object(),
    )

    assert selected_no_penalty.rf_max_depth == 6
    assert selected_with_penalty.rf_max_depth == 4


def test_select_computed_overlay_cfg_is_deterministic_under_ties(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    train_port = pd.Series(np.linspace(-0.01, 0.015, len(idx)), index=idx)

    def _flat_apply_hmm_softprob_rf_strategy(*, train_port_rets, test_port_rets, cfg, hmm_engine):
        wobble_train = 0.0001 * np.sin(np.linspace(0.0, 8.0, len(train_port_rets)))
        wobble_test = 0.0001 * np.sin(np.linspace(0.0, 8.0, len(test_port_rets)))
        ml_train = pd.Series(0.0010 + wobble_train, index=train_port_rets.index)
        ml_test = pd.Series(0.0010 + wobble_test, index=test_port_rets.index)
        diag = {"test_avg_exposure": float(cfg.ml_target_exposure)}
        return ml_train, ml_test, diag, pd.DataFrame(), {}

    monkeypatch.setattr(overlay_mod, "apply_hmm_softprob_rf_strategy", _flat_apply_hmm_softprob_rf_strategy)

    cfg = BacktestConfig(
        template_use_precomputed_artifacts=False,
        ml_computed_overlay_tuning_enabled=True,
        ml_computed_overlay_tuning_min_rows=60,
        ml_computed_overlay_tuning_folds=3,
        ml_computed_overlay_grid_rf_n_estimators=(300, 400),
        ml_computed_overlay_grid_rf_max_depth=(4, 6),
        ml_computed_overlay_grid_rf_min_samples_leaf=(20, 40),
        ml_computed_overlay_grid_target_exposure=(0.10, 0.40),
        ml_computed_overlay_grid_hmm_n_states=(2, 3),
    )

    selected_cfg_1, diag_1 = overlay_mod.select_computed_overlay_cfg(
        train_port_rets=train_port,
        cfg=cfg,
        hmm_engine=object(),
    )
    selected_cfg_2, diag_2 = overlay_mod.select_computed_overlay_cfg(
        train_port_rets=train_port,
        cfg=cfg,
        hmm_engine=object(),
    )

    assert diag_1["selection_score"] == diag_2["selection_score"]
    assert selected_cfg_1.rf_max_depth == selected_cfg_2.rf_max_depth == 4
    assert selected_cfg_1.rf_min_samples_leaf == selected_cfg_2.rf_min_samples_leaf == 40
    assert selected_cfg_1.ml_target_exposure == selected_cfg_2.ml_target_exposure == 0.10
    assert selected_cfg_1.hmm_n_states == selected_cfg_2.hmm_n_states == 2
    assert selected_cfg_1.rf_n_estimators == selected_cfg_2.rf_n_estimators == 300


def test_compute_template_ml_overlay_skips_tuning_in_precomputed_mode(monkeypatch):
    idx = pd.date_range("2021-01-01", periods=16, freq="D")
    train_df = pd.DataFrame({"A": np.linspace(-0.01, 0.01, len(idx))}, index=idx)
    test_df = pd.DataFrame({"A": np.linspace(-0.005, 0.012, len(idx))}, index=idx)
    weights = pd.Series({"A": 1.0})

    def _raise_if_called(**kwargs):
        raise AssertionError("select_computed_overlay_cfg should not be called in precomputed mode")

    def _fake_apply_hmm_softprob_rf_strategy(*, train_port_rets, test_port_rets, cfg, hmm_engine):
        return train_port_rets * 0.5, test_port_rets * 0.5, {}, pd.DataFrame(), {}

    monkeypatch.setattr(overlay_mod, "select_computed_overlay_cfg", _raise_if_called)
    monkeypatch.setattr(overlay_mod, "apply_hmm_softprob_rf_strategy", _fake_apply_hmm_softprob_rf_strategy)
    monkeypatch.setattr(overlay_mod, "compute_and_apply_ml_position_scalar", _passthrough_scalar)
    monkeypatch.setattr(overlay_mod, "compute_and_apply_vol_target", _passthrough_vol_target)

    cfg = BacktestConfig(template_use_precomputed_artifacts=True, ml_enabled=True)
    out = overlay_mod.compute_template_ml_overlay(
        train_returns_df=train_df,
        test_returns_df=test_df,
        weights=weights,
        cfg=cfg,
        hmm_engine=object(),
        include_series=False,
    )
    assert out["ml_tuning_diagnostics"]["applied"] is False


def test_compute_template_ml_overlay_uses_selected_cfg_in_computed_mode(monkeypatch):
    idx = pd.date_range("2022-01-01", periods=32, freq="D")
    train_df = pd.DataFrame({"A": np.linspace(-0.01, 0.01, len(idx))}, index=idx)
    test_df = pd.DataFrame({"A": np.linspace(-0.008, 0.012, len(idx))}, index=idx)
    weights = pd.Series({"A": 1.0})
    seen = {"rf_n_estimators": None}

    def _fake_select(*, train_port_rets, cfg, hmm_engine):
        tuned = replace(cfg, rf_n_estimators=999, ml_target_exposure=0.4, hmm_n_states=3, ml_auto_threshold=True)
        return tuned, {"applied": True, "selected_rf_n_estimators": 999}

    def _fake_apply_hmm_softprob_rf_strategy(*, train_port_rets, test_port_rets, cfg, hmm_engine):
        seen["rf_n_estimators"] = cfg.rf_n_estimators
        return train_port_rets * 0.7, test_port_rets * 0.7, {}, pd.DataFrame(), {}

    monkeypatch.setattr(overlay_mod, "select_computed_overlay_cfg", _fake_select)
    monkeypatch.setattr(overlay_mod, "apply_hmm_softprob_rf_strategy", _fake_apply_hmm_softprob_rf_strategy)
    monkeypatch.setattr(overlay_mod, "compute_and_apply_ml_position_scalar", _passthrough_scalar)
    monkeypatch.setattr(overlay_mod, "compute_and_apply_vol_target", _passthrough_vol_target)

    cfg = BacktestConfig(template_use_precomputed_artifacts=False, ml_enabled=True)
    out = overlay_mod.compute_template_ml_overlay(
        train_returns_df=train_df,
        test_returns_df=test_df,
        weights=weights,
        cfg=cfg,
        hmm_engine=object(),
        include_series=False,
    )

    assert seen["rf_n_estimators"] == 999
    assert out["ml_tuning_diagnostics"]["applied"] is True
