CI workflows

This repository now uses three GitHub Actions workflows:

- `.github/workflows/ci-fast.yml`  
  Push/PR fast gate with a targeted pytest subset (~seconds runtime) plus parity result threshold gate.
- `.github/workflows/validate_extension.yml`  
  PR/manual extension validation with deterministic gate checks and stress validation.
- `.github/workflows/ci-full-coverage.yml`  
  Nightly/manual near-full pytest run with coverage gate (`--cov-fail-under=70`).

## Local reproduction

Install CI dependencies:

```bash
python -m pip install -U pip
pip install pytest pytest-cov numpy pandas scipy scikit-learn optuna
export PYTHONPATH=src
```

Fast gate:

```bash
pytest -q \
  tests/test_run_pipeline_wiring.py \
  tests/test_pipeline_modular_data_signals.py \
  tests/test_ml_modular_path.py \
  tests/test_portfolio_allocation.py \
  tests/test_dd_guard.py \
  tests/test_validate_oos_gates.py \
  tests/test_phase4_online_learning.py \
  tests/test_phase5_stress_scenarios.py
python - <<'PY'
import json, pathlib
path = pathlib.Path("outputs_ci/parity-gate")
path.mkdir(parents=True, exist_ok=True)
with open(path / "parity_metrics_comparison.json", "w", encoding="utf-8") as f:
    json.dump({
        "run_a": {"sharpe": 1.00, "annual_return": 0.10, "annual_volatility": 0.11, "max_drawdown": -0.12},
        "run_b": {"sharpe": 1.08, "annual_return": 0.11, "annual_volatility": 0.12, "max_drawdown": -0.11},
        "diff": {"sharpe": 0.08, "annual_return": 0.01, "annual_volatility": 0.01, "max_drawdown": 0.01},
    }, f, indent=2)
PY
python scripts/evaluate_parity_gate.py outputs_ci/parity-gate/parity_metrics_comparison.json
```

Deterministic + extension validation:

```bash
pytest -q tests/test_validate_oos_gates.py tests/test_bootstrap_validation.py
pytest -q tests/test_phase1_bayesian_hyperopt.py tests/test_phase1_dynamic_ensemble.py tests/test_phase4_feature_interactions.py
python scripts/run_stress_scenarios.py --run-id ci-stress-validation --out-root outputs_ci
```

Near-full coverage gate:

```bash
pytest -q tests \
  --ignore=tests/test_parity_wave3.py \
  --ignore=tests/test_runner_orchestration.py \
  --cov=src/tema \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml:coverage.xml \
  --cov-fail-under=70
```
