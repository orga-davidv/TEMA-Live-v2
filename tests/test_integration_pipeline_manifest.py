import json
import os
import pandas as pd
from tema.pipeline.runner import run_pipeline
from tema.config import BacktestConfig
from tema.validation.manifest import load_manifest_schema, validate_manifest_schema


def make_csv(filepath, rows=10):
    idx = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    df = pd.DataFrame({"datetime": idx, "close": [1.0 + float(i) * 0.01 for i in range(rows)]})
    df.to_csv(filepath, index=False)


def test_run_pipeline_writes_manifest_and_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # create 2 assets with 6 rows (below default min to force using min_rows override)
    make_csv(data_dir / "asset1_merged.csv", rows=6)
    make_csv(data_dir / "asset2_merged.csv", rows=6)

    cfg = BacktestConfig()
    cfg.modular_data_signals_enabled = True
    cfg.data_path = str(data_dir)
    cfg.data_min_rows = 3
    cfg.data_max_assets = 2
    cfg.portfolio_modular_enabled = True
    cfg.ml_modular_path_enabled = True
    cfg.vol_target_enabled = True
    cfg.signal_use_cpp = False

    out = run_pipeline(run_id="integration-test-1", cfg=cfg, out_root=str(tmp_path))
    manifest_path = out.get("manifest_path")
    assert manifest_path and os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    schema = load_manifest_schema()
    ok, errors = validate_manifest_schema(manifest, schema=schema)
    assert ok, f"manifest schema validation failed: {errors}"
    # manifest should list artifacts and run_id
    assert manifest.get("run_id") == "integration-test-1"
    artifacts = manifest.get("artifacts", [])
    # ensure artifact files exist on disk
    out_dir = out.get("out_dir")
    for a in artifacts:
        p = os.path.join(out_dir, f"{a}.json")
        assert os.path.exists(p)


if __name__ == '__main__':
    test_run_pipeline_writes_manifest_and_artifacts().__call__()
