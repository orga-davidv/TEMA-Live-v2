"""
Simple runner to execute turnover ablation scenarios for TEMA-TEMPLATE(NEW_).py
The runner launches the template script with a small set of environment overrides,
collects the produced CSV metrics (bl_portfolio_metrics.csv and ml_diagnostics.csv)
and writes a consolidated turnover_ablation_results.csv in the same folder.

Usage:
    python3 turnover_ablation.py

No editing of the template source is required - the template reads these env vars
when present:
  REB_MIN_THRESHOLD
  COST_AWARE_REBALANCE
  COST_AWARE_REBALANCE_MULTIPLIER
  TURNOVER_PENALTY_LAMBDA

Keep runs isolated: each scenario is run in a fresh subprocess and overwrites
the Template/ CSV outputs; the runner captures them after each run.
"""
from pathlib import Path
import os
import subprocess
import time
import pandas as pd
import sys

HERE = Path(__file__).resolve().parent
TEMPLATE_SCRIPT = HERE / "TEMA-TEMPLATE(NEW_).py"

SCENARIOS = {
    "baseline_none": {
        "REB_MIN_THRESHOLD": "0.0",
        "COST_AWARE_REBALANCE": "0",
        "COST_AWARE_REBALANCE_MULTIPLIER": "1.0",
        "TURNOVER_PENALTY_LAMBDA": "0.0",
    },
    "mech1_only": {
        "REB_MIN_THRESHOLD": "0.01",
        "COST_AWARE_REBALANCE": "0",
        "COST_AWARE_REBALANCE_MULTIPLIER": "1.0",
        "TURNOVER_PENALTY_LAMBDA": "0.0",
    },
    "mech3_only": {
        "REB_MIN_THRESHOLD": "0.0",
        "COST_AWARE_REBALANCE": "0",
        "COST_AWARE_REBALANCE_MULTIPLIER": "1.0",
        # reasonable non-zero penalty for testing
        "TURNOVER_PENALTY_LAMBDA": "0.5",
    },
    "mech1_plus_2": {
        "REB_MIN_THRESHOLD": "0.01",
        "COST_AWARE_REBALANCE": "1",
        "COST_AWARE_REBALANCE_MULTIPLIER": "1.0",
        "TURNOVER_PENALTY_LAMBDA": "0.0",
    },
    "all_three": {
        "REB_MIN_THRESHOLD": "0.01",
        "COST_AWARE_REBALANCE": "1",
        "COST_AWARE_REBALANCE_MULTIPLIER": "1.0",
        "TURNOVER_PENALTY_LAMBDA": "0.5",
    },
}

OUT_PATH = HERE / "turnover_ablation_results.csv"
BL_METRICS = HERE / "bl_portfolio_metrics.csv"
ML_DIAG = HERE / "ml_diagnostics.csv"


def run_scenario(name: str, env_overrides: dict) -> dict:
    print(f"Running scenario: {name}")
    env = os.environ.copy()
    env.update(env_overrides)

    # Remove possibly stale outputs before run
    for p in (BL_METRICS, ML_DIAG):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    cmd = [sys.executable, str(TEMPLATE_SCRIPT)]
    start = time.time()
    try:
        subprocess.run(cmd, cwd=str(HERE), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Template script failed for scenario {name}: {exc}")
        return {"scenario": name, "error": str(exc)}
    elapsed = time.time() - start
    print(f"Scenario {name} finished in {elapsed:.1f}s")

    out: dict = {"scenario": name, "runtime_s": elapsed}

    # Read BL metrics
    if BL_METRICS.exists():
        try:
            df = pd.read_csv(BL_METRICS)
            # pick train and test rows if present
            for ds in ("train", "test", "train_ml", "test_ml"):
                row = df[df.get("dataset") == ds]
                if not row.empty:
                    prefix = f"bl_{ds}"
                    r = row.iloc[0].to_dict()
                    for k, v in r.items():
                        if k == "dataset":
                            continue
                        out[f"{prefix}_{k}"] = v
        except Exception as exc:
            out["bl_read_error"] = str(exc)
    else:
        out["bl_missing"] = True

    # Read ML diagnostics (first row flattened)
    if ML_DIAG.exists():
        try:
            dfm = pd.read_csv(ML_DIAG)
            if not dfm.empty:
                m = dfm.iloc[0].to_dict()
                for k, v in m.items():
                    out[f"ml_{k}"] = v
        except Exception as exc:
            out["ml_read_error"] = str(exc)
    else:
        out["ml_missing"] = True

    return out


def main():
    results = []
    for name, envs in SCENARIOS.items():
        row = run_scenario(name, envs)
        results.append(row)

    # Normalize to dataframe and write
    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved consolidated results to {OUT_PATH}")


if __name__ == "__main__":
    main()
