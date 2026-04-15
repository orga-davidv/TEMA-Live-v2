import json
import os
import subprocess
from pathlib import Path
from tema.pipeline import run_pipeline


def test_modular_runner_creates_manifest(tmp_path):
    out_root = str(tmp_path / "outputs")
    run_id = "test-modular"
    res = run_pipeline(run_id=run_id, cfg=None, out_root=out_root)
    mf_path = Path(res["manifest_path"])
    assert mf_path.exists(), f"manifest not found: {mf_path}"
    data = json.loads(mf_path.read_text())
    assert data.get("run_id") == run_id
    assert "artifacts" in data


def test_legacy_fallback_creates_manifest(tmp_path):
    out_root = str(tmp_path / "outputs")
    run_id = "test-legacy"
    # Call the CLI script to exercise legacy path
    cwd = Path(__file__).resolve().parent.parent
    cmd = ["python", str(cwd / 'run_pipeline.py'), "--legacy", "--run-id", run_id]
    subprocess.check_call(cmd, cwd=str(cwd))
    mf = Path(out_root) / run_id / "manifest.json"
    # In the CLI we write manifest under project outputs/ by default; for test we didn't override OUT_ROOT
    # So check in cwd/outputs
    mf = Path(str(cwd)) / "outputs" / run_id / "manifest.json"
    assert mf.exists(), f"legacy manifest missing: {mf}"
    data = json.loads(mf.read_text())
    # By default the CLI writes a best-effort manifest and will only execute the
    # legacy monolith if the environment variable TEMA_RUN_LEGACY_EXECUTE=1 is set.
    assert "legacy_executed" in data
