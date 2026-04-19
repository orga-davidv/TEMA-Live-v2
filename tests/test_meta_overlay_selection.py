import numpy as np
import pandas as pd

from tema.config import BacktestConfig
from tema.ml.meta_overlay import compute_ml_meta_overlay_series


def _synthetic_series():
    rng = np.random.default_rng(42)
    idx_train = pd.date_range("2020-01-01", periods=320, freq="D")
    idx_test = pd.date_range("2021-01-01", periods=140, freq="D")

    baseline_train = pd.Series(rng.normal(0.0, 0.01, len(idx_train)), index=idx_train)
    baseline_test = pd.Series(rng.normal(0.0, 0.01, len(idx_test)), index=idx_test)

    ml_train = pd.Series(
        0.0005 * np.sign(np.sin(np.arange(len(idx_train)) / 8.0)) + rng.normal(0.0, 0.0015, len(idx_train)),
        index=idx_train,
    )
    ml_test = pd.Series(
        0.0005 * np.sign(np.sin(np.arange(len(idx_test)) / 8.0)) + rng.normal(0.0, 0.0015, len(idx_test)),
        index=idx_test,
    )
    return baseline_train, baseline_test, ml_train, ml_test


def test_meta_overlay_computed_mode_uses_regularized_selection():
    baseline_train, baseline_test, ml_train, ml_test = _synthetic_series()
    cfg = BacktestConfig(template_use_precomputed_artifacts=False, ml_meta_overlay_enabled=True)

    expo_train, expo_test, meta_train, meta_test, diag = compute_ml_meta_overlay_series(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        ml_train=ml_train,
        ml_test=ml_test,
        cfg=cfg,
    )

    assert diag["selection_mode"] == "computed_regularized"
    assert diag["selection"].get("folds_evaluated", 0) >= 1
    assert len(expo_train) == len(meta_train)
    assert len(expo_test) == len(meta_test)


def test_meta_overlay_computed_mode_selection_is_deterministic():
    baseline_train, baseline_test, ml_train, ml_test = _synthetic_series()
    cfg = BacktestConfig(
        template_use_precomputed_artifacts=False,
        ml_meta_overlay_enabled=True,
        ml_meta_computed_validation_folds=3,
        ml_meta_computed_overfit_penalty=0.5,
    )

    run_1 = compute_ml_meta_overlay_series(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        ml_train=ml_train,
        ml_test=ml_test,
        cfg=cfg,
    )
    run_2 = compute_ml_meta_overlay_series(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        ml_train=ml_train,
        ml_test=ml_test,
        cfg=cfg,
    )

    *_series_1, diag_1 = run_1
    *_series_2, diag_2 = run_2
    assert diag_1["chosen_method"] == diag_2["chosen_method"]
    assert diag_1["chosen_k"] == diag_2["chosen_k"]
    assert diag_1["chosen_floor"] == diag_2["chosen_floor"]


def test_meta_overlay_template_mode_stays_template_compatible():
    baseline_train, baseline_test, ml_train, ml_test = _synthetic_series()
    cfg = BacktestConfig(template_use_precomputed_artifacts=True, ml_meta_overlay_enabled=True)

    *_series, diag = compute_ml_meta_overlay_series(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        ml_train=ml_train,
        ml_test=ml_test,
        cfg=cfg,
    )

    assert diag["selection_mode"] == "template_compatible"
    assert diag["chosen_floor"] >= 0.2
