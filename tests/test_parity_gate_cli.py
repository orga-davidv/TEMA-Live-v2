import json
import os
import subprocess
import sys


def test_evaluate_parity_gate_cli_exit_codes(tmp_path):
    # Diffs selected to be within the stricter default parity thresholds
    comparison = {
        "run_a": {"sharpe": 1.0, "annual_return": 0.10, "annual_volatility": 0.10, "max_drawdown": -0.10},
        "run_b": {"sharpe": 1.01, "annual_return": 0.103, "annual_volatility": 0.105, "max_drawdown": -0.11},
        "diff": {"sharpe": 0.01, "annual_return": 0.003, "annual_volatility": 0.005, "max_drawdown": -0.01},
    }
    comp_path = tmp_path / "parity_metrics_comparison.json"
    comp_path.write_text(json.dumps(comparison), encoding="utf-8")

    cmd = [sys.executable, os.path.join("scripts", "evaluate_parity_gate.py"), str(comp_path)]
    ok = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert ok.returncode == 0

    bad = subprocess.run(
        cmd + ["--threshold-sharpe", "0.001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert bad.returncode == 1
