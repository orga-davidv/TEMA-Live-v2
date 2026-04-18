import os
import shutil
import json
import tempfile
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import run_pipeline
from tema import validation


def test_modular_and_legacy_manifests(tmp_path, monkeypatch):
    # Use deterministic run ids and outputs under tmp_path
    run_id_mod = "test-modular"
    run_id_leg = "test-legacy"
    out_root = str(tmp_path / "outputs")

    # Run modular pipeline
    res_mod = run_pipeline.run_modular(run_id=run_id_mod, out_root=out_root)
    assert "manifest_path" in res_mod
    mod_mf = res_mod["manifest_path"]
    assert os.path.exists(mod_mf)

    # Run legacy fallback (should NOT execute legacy by default)
    # Ensure env var not set
    monkeypatch.delenv("TEMA_RUN_LEGACY_EXECUTE", raising=False)
    res_leg = run_pipeline.run_legacy(run_id=run_id_leg, out_root=out_root)
    leg_mf = res_leg["manifest_path"]
    assert os.path.exists(leg_mf)

    # Compare using validation helper
    report = validation.compare_manifests(mod_manifest_path=mod_mf, legacy_manifest_path=leg_mf)

    # Manifest keys check should pass for modular runner
    assert report["results"]["manifest_keys"]["ok"] is True

    # Legacy status fields should be present (legacy_executed flag)
    assert report["results"]["legacy_status_fields"]["ok"] is True

    # Artifacts should exist for modular run
    assert report["results"]["artifact_presence"]["ok"] is True

    # run_id fields should be recorded (they may differ if caller used different ids)
    assert "mod_run_id" in report["results"]["run_id_match"]
    assert "leg_run_id" in report["results"]["run_id_match"]


def test_parity_helper_smoke(tmp_path):
    # Sanity check that the helper loads manifests and reports structure
    # Create two minimal manifests
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    a_manifest = {"schema_version": "manifest.v1", "run_id": "a1", "timestamp": "t", "artifacts": ["x"]}
    b_manifest = {"run_id": "a1", "legacy_executed": False}

    a_path = a_dir / "manifest.json"
    b_path = b_dir / "manifest.json"
    a_path.write_text(json.dumps(a_manifest))
    b_path.write_text(json.dumps(b_manifest))

    report = validation.compare_manifests(str(a_path), str(b_path))
    assert report["results"]["manifest_keys"]["ok"] is True
    assert report["results"]["legacy_status_fields"]["ok"] is True
