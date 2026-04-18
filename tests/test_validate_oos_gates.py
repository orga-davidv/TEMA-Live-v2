import json
import os
import subprocess
import sys
import tempfile

from tema.validation.oos import validate_oos_gates


def _write_manifest_and_perf(tmpdir, artifacts: dict):
    manifest = {"run_id": "r1", "timestamp": "2026-01-01T00:00:00Z", "artifacts": ["perf"]}
    manifest_path = os.path.join(tmpdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    perf_path = os.path.join(tmpdir, "perf.json")
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f)
    return manifest_path, perf_path


def test_validate_pass_and_cli_exit_code(tmp_path):
    tmpdir = str(tmp_path)
    artifacts = {"sharpe": 1.5, "max_drawdown": -0.05, "annualized_turnover": 0.2, "stress_scenarios": {"s1": {"drawdown": -0.03}}}
    manifest_path, perf_path = _write_manifest_and_perf(tmpdir, artifacts)

    res = validate_oos_gates(manifest_path, min_sharpe=1.0, max_drawdown=0.1, max_turnover=0.5)
    assert res["passed"] is True

    # CLI invocation should return exit code 0
    cmd = [sys.executable, os.path.join("scripts", "validate_oos_gates.py"), tmpdir, "--min-sharpe", "1.0", "--max-drawdown", "0.1", "--max-turnover", "0.5"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode == 0
    out = json.loads(p.stdout.decode())
    assert out.get("passed") is True


def test_validate_fail_on_sharpe(tmp_path):
    tmpdir = str(tmp_path)
    artifacts = {"sharpe": 0.5, "max_drawdown": -0.01, "annualized_turnover": 0.1}
    manifest_path, perf_path = _write_manifest_and_perf(tmpdir, artifacts)

    res = validate_oos_gates(manifest_path, min_sharpe=1.0)
    assert res["passed"] is False

    # CLI should return non-zero
    cmd = [sys.executable, os.path.join("scripts", "validate_oos_gates.py"), tmpdir, "--min-sharpe", "1.0"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode != 0
    out = json.loads(p.stdout.decode())
    assert out.get("passed") is False


def test_validate_missing_metrics_is_graceful(tmp_path):
    tmpdir = str(tmp_path)
    # Manifest points to an artifact without recognizable perf metrics.
    manifest = {"run_id": "r2", "timestamp": "2026-01-01T00:00:00Z", "artifacts": ["misc"]}
    manifest_path = os.path.join(tmpdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(tmpdir, "misc.json"), "w", encoding="utf-8") as f:
        json.dump({"foo": 1, "bar": "x"}, f)

    res = validate_oos_gates(manifest_path, min_sharpe=1.0, max_drawdown=0.1, max_turnover=0.5)
    assert res["passed"] is True
    assert res["checks"]["artifact_metrics_found"]["skipped"] is True

    cmd = [
        sys.executable,
        os.path.join("scripts", "validate_oos_gates.py"),
        tmpdir,
        "--min-sharpe",
        "1.0",
        "--max-drawdown",
        "0.1",
        "--max-turnover",
        "0.5",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode == 0
