import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

MANIFEST_SCHEMA_VERSION = "manifest.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_manifest_schema(schema_version: str = MANIFEST_SCHEMA_VERSION) -> Dict:
    schema_path = _repo_root() / "schemas" / f"{schema_version}.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_json_schema(instance, schema: Dict, path: str, errors: List[str]) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(instance, dict):
            errors.append(f"{path}: expected object")
            return
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}.{key}: missing required property")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                _validate_json_schema(value, properties[key], f"{path}.{key}", errors)
    elif expected_type == "array":
        if not isinstance(instance, list):
            errors.append(f"{path}: expected array")
            return
        if schema.get("uniqueItems", False):
            seen = set()
            for idx, item in enumerate(instance):
                marker = json.dumps(item, sort_keys=True)
                if marker in seen:
                    errors.append(f"{path}[{idx}]: duplicate item")
                seen.add(marker)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(instance):
                _validate_json_schema(item, item_schema, f"{path}[{idx}]", errors)
    elif expected_type == "string":
        if not isinstance(instance, str):
            errors.append(f"{path}: expected string")
            return
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.match(pattern, instance) is None:
            errors.append(f"{path}: does not match pattern {pattern}")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{path}: invalid date-time")
    elif expected_type == "boolean":
        if not isinstance(instance, bool):
            errors.append(f"{path}: expected boolean")
    elif expected_type == "number":
        if not isinstance(instance, (int, float)) or isinstance(instance, bool):
            errors.append(f"{path}: expected number")
    elif expected_type == "integer":
        if not isinstance(instance, int) or isinstance(instance, bool):
            errors.append(f"{path}: expected integer")

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected constant value {schema['const']}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']}")


def validate_manifest_schema(manifest: Dict, schema: Dict | None = None) -> Tuple[bool, List[str]]:
    schema_obj = schema or load_manifest_schema()
    errors: List[str] = []
    _validate_json_schema(manifest, schema_obj, "$", errors)
    return (len(errors) == 0, errors)


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
    """Compare modular manifest with legacy_manifest and return a structured report.

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
    req_keys = ["schema_version", "run_id", "timestamp", "artifacts"]
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
