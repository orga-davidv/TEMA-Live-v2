CI (fast checks)

This repository includes a lightweight GitHub Actions workflow at .github/workflows/ci-fast.yml.

Commands used locally to reproduce the CI smoke checks:

- Install project and test runner:

  python -m pip install -U pip
  pip install -e . pytest

- Run the targeted fast test subset used by CI:

  pytest -q tests/test_config_load.py tests/test_pipeline_runner.py tests/test_pipeline_cli.py tests/test_ml_refactor.py tests/test_risk_budget.py tests/test_portfolio_black_litterman.py tests/test_rebalancer.py tests/test_vol_target.py tests/test_parity_regression.py -q

Notes:
- The CI intentionally focuses on a fast subset of tests covering core modules: config, pipeline, signals/ml, portfolio, scaling, risk, reporting/parity.
- If you add expensive or generated artifacts under outputs/ make sure to exclude smoke folders or add a caching strategy.
