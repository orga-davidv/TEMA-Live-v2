import json
import os
from pathlib import Path

from parity_compare import find_metrics_in_artifact, compare_runs


def test_find_metrics_in_artifact():
    data = {
        "sharpe": 1.23,
        "annual_return": 0.10,
        "annual_volatility": 0.08,
        "max_drawdown": -0.12,
    }
    res = find_metrics_in_artifact(data)
    assert res["sharpe"] == 1.23
    assert res["annual_return"] == 0.10
    assert res["annual_volatility"] == 0.08
    assert res["max_drawdown"] == -0.12


def test_compare_runs(tmp_path):
    # create two run dirs with manifest and a performance.json
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    (a_dir / "manifest.json").write_text(json.dumps({"artifacts": ["performance"]}))
    (b_dir / "manifest.json").write_text(json.dumps({"artifacts": ["performance"]}))

    perf_a = {"sharpe": 0.5, "annual_return": 0.05}
    perf_b = {"sharpe": 1.0, "annual_return": 0.12}

    (a_dir / "performance.json").write_text(json.dumps(perf_a))
    (b_dir / "performance.json").write_text(json.dumps(perf_b))

    comp = compare_runs(str(a_dir / "manifest.json"), str(b_dir / "manifest.json"))
    assert comp["run_a"]["sharpe"] == 0.5
    assert comp["run_b"]["sharpe"] == 1.0
    assert comp["diff"]["sharpe"] == 0.5
    # annual_return diff
    assert abs(comp["diff"]["annual_return"] - (0.12 - 0.05)) < 1e-12
