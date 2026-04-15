import json
import os
from typing import Dict, List, Tuple


def load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_manifest_keys(manifest: Dict, required_keys: List[str]) -> Tuple[bool, List[str]]:
    missing = [k for k in required_keys if k not in manifest]
    return (len(missing) == 0, missing)


def check_artifacts_exist(manifest: Dict, out_dir: str) -> Tuple[bool, List[str]]:
    """Verify that every artifact listed in manifest['artifacts'] has a corresponding file in out_dir.

    Returns (all_present, missing_files)
    """
    artifacts = manifest.get("artifacts", [])
    missing = []
    for a in artifacts:
        p = os.path.join(out_dir, f"{a}.json")
        if not os.path.exists(p):
            missing.append(p)
    return (len(missing) == 0, missing)


def compare_manifests(mod_manifest_path: str, legacy_manifest_path: str) -> Dict:
    """Compare modular manifest with legacy manifest and return a structured report.

    Checks performed:
    - manifest required keys (modular)
    - legacy status fields (legacy_executed)
    - artifact presence for modular run

    The function is intentionally conservative and returns details for downstream tests.
    """
    report = {
        "modular": {"path": mod_manifest_path},
        "legacy": {"path": legacy_manifest_path},
        "results": {},
    }

    mod = load_manifest(mod_manifest_path)
    leg = load_manifest(legacy_manifest_path)

    # 1) Modular manifest required keys
    req_keys = ["run_id", "timestamp", "artifacts"]
    ok_keys, missing = check_manifest_keys(mod, req_keys)
    report["results"]["manifest_keys"] = {"ok": ok_keys, "missing": missing}

    # 2) Legacy status fields
    # Legacy manifest created by run_pipeline.run_legacy should contain 'run_id' and a boolean
    # 'legacy_executed' indicating whether the monolith actually ran.
    legacy_status_ok = "legacy_executed" in leg
    report["results"]["legacy_status_fields"] = {"ok": legacy_status_ok, "present": [k for k in ["legacy_executed"] if k in leg]}

    # 3) Artifact presence expectations
    out_dir = os.path.dirname(mod_manifest_path)
    artifacts_ok, missing_files = check_artifacts_exist(mod, out_dir)
    report["results"]["artifact_presence"] = {"ok": artifacts_ok, "missing_files": missing_files}

    # 4) Basic run_id parity
    report["results"]["run_id_match"] = {"ok": mod.get("run_id") == leg.get("run_id"), "mod_run_id": mod.get("run_id"), "leg_run_id": leg.get("run_id")}

    return report
