"""Small CLI to run either the modular Wave 2 pipeline or the legacy monolith.

Usage:
  python run_pipeline.py [--run-id ID] [--legacy]

If --legacy is provided we execute Template/TEMA-TEMPLATE(NEW_).py via runpy.
Otherwise we call src.tema.pipeline.run_pipeline.

This script is intentionally minimal and deterministic so CI/tests can exercise both
paths without touching project-wide configuration.
"""
import argparse
import sys
import os
import runpy
from pathlib import Path
import re
import json
import csv

ROOT = Path(__file__).resolve().parent
# Ensure src is on sys.path so "tema" package can be imported
sys.path.insert(0, str(ROOT / "src"))


def run_legacy(run_id: str, out_root: str = "outputs"):
    """Run the legacy monolith only when the env var TEMA_RUN_LEGACY_EXECUTE=1 is set.

    By default this function will create a best-effort manifest and NOT execute the
    legacy script. This keeps the CLI safe and deterministic for CI/tests while still
    providing an explicit opt-in to run the old monolith.
    """
    legacy_path = ROOT / "Template" / "TEMA-TEMPLATE(NEW_).py"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy monolith not found: {legacy_path}")

    should_exec = os.environ.get("TEMA_RUN_LEGACY_EXECUTE", "0") == "1"
    # sanitize run_id to avoid path traversal
    if not re.match(r'^[A-Za-z0-9._-]+$', run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")

    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mf = out_dir / "manifest.json"
    metrics_dataset = os.environ.get("TEMA_LEGACY_METRICS_DATASET", "test")

    def _write_manifest(extra: dict | None = None):
        payload = {"run_id": run_id}
        if extra:
            payload.update(extra)
        with open(mf, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)

    def _extract_legacy_performance() -> dict | None:
        metrics_csv = legacy_path.parent / "bl_portfolio_metrics.csv"
        if not metrics_csv.exists():
            return None
        with open(metrics_csv, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return None
        row = next((r for r in rows if str(r.get("dataset", "")).strip().lower() == metrics_dataset.lower()), rows[0])
        def _f(name: str):
            v = row.get(name)
            if v is None or v == "":
                return None
            return float(v)
        return {
            "sharpe": _f("sharpe_ratio"),
            "annual_return": _f("annualized_return"),
            "annual_volatility": _f("annualized_volatility"),
            "max_drawdown": _f("max_drawdown"),
            "legacy_dataset": row.get("dataset"),
            "legacy_metrics_source": str(metrics_csv),
        }
    if should_exec:
        # run in its own globals to emulate script execution
        g = {"__name__": "__main__", "RUN_ID": run_id, "OUT_ROOT": out_root}
        runpy.run_path(str(legacy_path), run_name="__main__", init_globals=g)
        perf = _extract_legacy_performance()
        if perf is not None:
            perf_path = out_dir / "performance.json"
            with open(perf_path, "w", encoding="utf-8") as fh:
                json.dump(perf, fh, indent=2)
            _write_manifest({"legacy_executed": True, "artifacts": ["performance"]})
        else:
            _write_manifest({"legacy_executed": True, "note": "legacy metrics CSV not found after execution"})
    else:
        # do not execute by default; record that we skipped execution
        _write_manifest({"legacy_executed": False, "note": "set TEMA_RUN_LEGACY_EXECUTE=1 to actually run the legacy script"})

    return {'manifest_path': str(mf), 'out_dir': str(out_dir)}


def run_modular(
    run_id: str,
    out_root: str = "outputs",
    stress_enabled: bool = False,
    stress_seed: int = 42,
    stress_n_paths: int = 200,
    stress_horizon: int = 20,
    modular_data_signals_enabled: bool = False,
    modular_portfolio_enabled: bool = False,
    data_path: str | None = None,
    ml_enabled: bool = True,
    ml_modular_path_enabled: bool = False,
    ml_probability_threshold: float = 0.0,
    data_max_assets: int = 3,
    data_full_universe_for_parity: bool = True,
    portfolio_method: str = "bl",
    portfolio_risk_aversion: float = 2.5,
    portfolio_bl_tau: float = 0.05,
    portfolio_bl_view_confidence: float = 0.65,
    ml_hmm_scalar_floor: float = 0.30,
    ml_hmm_scalar_ceiling: float = 1.50,
    vol_target_apply_to_ml: bool = False,
):
    from tema.pipeline import run_pipeline as rp
    from tema.config import BacktestConfig

    cfg = BacktestConfig(
        stress_enabled=stress_enabled,
        stress_seed=stress_seed,
        stress_n_paths=stress_n_paths,
        stress_horizon=stress_horizon,
        modular_data_signals_enabled=modular_data_signals_enabled,
        portfolio_modular_enabled=modular_portfolio_enabled,
        data_path=data_path,
        ml_enabled=ml_enabled,
        ml_modular_path_enabled=ml_modular_path_enabled,
        ml_probability_threshold=ml_probability_threshold,
        data_max_assets=data_max_assets,
        data_full_universe_for_parity=data_full_universe_for_parity,
        portfolio_method=portfolio_method,
        portfolio_risk_aversion=portfolio_risk_aversion,
        portfolio_bl_tau=portfolio_bl_tau,
        portfolio_bl_view_confidence=portfolio_bl_view_confidence,
        ml_hmm_scalar_floor=ml_hmm_scalar_floor,
        ml_hmm_scalar_ceiling=ml_hmm_scalar_ceiling,
        vol_target_apply_to_ml=vol_target_apply_to_ml,
    )
    return rp(run_id=run_id, cfg=cfg, out_root=out_root)


def main(argv=None):
    p = argparse.ArgumentParser("run_pipeline")
    p.add_argument("--run-id", default="manual-run")
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--stress-enabled", action="store_true")
    p.add_argument("--stress-seed", type=int, default=42)
    p.add_argument("--stress-n-paths", type=int, default=200)
    p.add_argument("--stress-horizon", type=int, default=20)
    p.add_argument("--modular-data-signals", action="store_true")
    p.add_argument("--modular-portfolio", action="store_true")
    p.add_argument("--data-path", default=None)
    p.add_argument("--ml-disabled", action="store_true")
    p.add_argument("--ml-modular-path", action="store_true")
    p.add_argument("--ml-prob-threshold", type=float, default=0.0)
    p.add_argument("--data-max-assets", type=int, default=3)
    p.add_argument("--disable-full-universe-override", action="store_true")
    p.add_argument("--portfolio-method", default="bl")
    p.add_argument("--portfolio-risk-aversion", type=float, default=2.5)
    p.add_argument("--portfolio-bl-tau", type=float, default=0.05)
    p.add_argument("--portfolio-view-confidence", type=float, default=0.65)
    p.add_argument("--ml-hmm-scalar-floor", type=float, default=0.30)
    p.add_argument("--ml-hmm-scalar-ceiling", type=float, default=1.50)
    p.add_argument("--vol-target-apply-to-ml", action="store_true")
    args = p.parse_args(argv)

    if args.legacy:
        res = run_legacy(args.run_id)
    else:
        res = run_modular(
            args.run_id,
            stress_enabled=args.stress_enabled,
            stress_seed=args.stress_seed,
            stress_n_paths=args.stress_n_paths,
            stress_horizon=args.stress_horizon,
            modular_data_signals_enabled=args.modular_data_signals,
            modular_portfolio_enabled=args.modular_portfolio,
            data_path=args.data_path,
            ml_enabled=(not args.ml_disabled),
            ml_modular_path_enabled=args.ml_modular_path,
            ml_probability_threshold=args.ml_prob_threshold,
            data_max_assets=args.data_max_assets,
            data_full_universe_for_parity=(not args.disable_full_universe_override),
            portfolio_method=args.portfolio_method,
            portfolio_risk_aversion=args.portfolio_risk_aversion,
            portfolio_bl_tau=args.portfolio_bl_tau,
            portfolio_bl_view_confidence=args.portfolio_view_confidence,
            ml_hmm_scalar_floor=args.ml_hmm_scalar_floor,
            ml_hmm_scalar_ceiling=args.ml_hmm_scalar_ceiling,
            vol_target_apply_to_ml=args.vol_target_apply_to_ml,
        )
    print(res)
    return res


if __name__ == "__main__":
    main()
