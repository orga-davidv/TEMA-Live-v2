# Parity Final Acceptance Summary

Run ID: final-parity-pass
Date: 2026-04-15

Summary:
- Executed final parity acceptance run using scripts/parity_harness.py with --execute-legacy and --fast flags.
- Produced outputs/final-parity-pass/parity_metrics_comparison.json and .csv.

Key metrics (modular run = run_a, legacy run = run_b):
- sharpe: run_a = -0.0140, run_b = 1.0274, diff (b - a) = 1.0414
- annual_return: run_a = -0.00544, run_b = 0.10325, diff = 0.10869
- annual_volatility: run_a = 0.09131, run_b = 0.10050, diff = 0.00919
- max_drawdown: run_a = -0.15070, run_b = -0.13509, diff = 0.01562

Interpretation / Remaining gap:
- The modular run is materially different from legacy on Sharpe and annual return (large positive delta favoring legacy in this configuration). Volatility and max drawdown differences are small but non-negligible.
- This indicates parity is not achieved for the full pipeline in the current configuration. Primary suspected gap areas (from MERGE_GAP_AUDIT) remain: out-of-sample combo selection (OOS), cost-aware rebalancing gates, and asset-level orchestration.

Artifacts written (intentional):
- outputs/final-parity-pass/parity_metrics_comparison.json
- outputs/final-parity-pass/parity_metrics_comparison.csv

Notes on working tree cleanliness:
- Committed parity-related files: scripts/parity_harness.py, src/parity_compare.py, tests/test_parity_compare.py.
- The outputs/ directory is generated and intentionally left untracked.
- There remain other modified files in the working tree (Template/*, run_pipeline.py, src/tema/*, tests/*). These were present prior to this acceptance pass and were not modified as part of the parity commits.

Next steps:
- Investigate OOS combo selection and cost-aware gating gaps identified in MERGE_GAP_AUDIT.
- Create targeted extraction tasks for OOS logic and gating and add unit/integration tests that exercise those paths.
- Re-run parity harness after implementing fixes and aim for diffs within acceptable thresholds (to be defined).

Status: partial — parity harness executed and artifacts produced, but parity not achieved.
