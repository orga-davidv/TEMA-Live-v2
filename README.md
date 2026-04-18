# TEMA-Live-v2

Modular TEMA pipeline with optional legacy fallback runner.

## Quickstart

```bash
python -m pip install -U pip
pip install -e . pytest
python run_pipeline.py --run-id smoke
```

## Output structure

Runs write to `outputs/<run_id>/` (or `--out-root <path>/<run_id>/`):

```text
outputs/<run_id>/
  manifest.json
  performance.json
  final_weights.json
  returns_csv_info.json
  ...
```

`manifest.json` includes `schema_version`, `run_id`, `timestamp`, and declared `artifacts`.

## Key flags matrix

| Flag | Default | Purpose |
|---|---:|---|
| `--run-id <id>` | `manual-run` | Output folder name under `outputs/`. |
| `--out-root <dir>` | `outputs` | Change output root directory. |
| `--legacy` | off | Use legacy path (executes only with `TEMA_RUN_LEGACY_EXECUTE=1`). |
| `--stress-enabled` | off | Add stress-scenario artifact generation. |
| `--modular-data-signals` | off | Force modular data/signal pipeline path. |
| `--modular-portfolio` | off | Force modular portfolio allocation path. |
| `--ml-disabled` | off | Disable ML scaling path. |
| `--ml-modular-path` | off | Enable modular ML probability/scalar path. |
| `--template-default-universe` | off | Template-like data/signal defaults. |
| `--no-default-validation-suite` | off | Skip default WF/OOS/bootstrap/MC validation bundle. |
| `--no-validation-graphs` | off | Disable validation PNG chart outputs. |
