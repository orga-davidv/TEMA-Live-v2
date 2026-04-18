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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Ensure src is on sys.path so "tema" package can be imported
sys.path.insert(0, str(ROOT / "src"))
from tema.validation.manifest import MANIFEST_SCHEMA_VERSION


def _extract_legacy_performance(
    metrics_csv: Path, metrics_dataset: str, require_dataset_match: bool = False
) -> dict | None:
    if not metrics_csv.exists():
        return None
    with open(metrics_csv, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = next(
        (r for r in rows if str(r.get("dataset", "")).strip().lower() == metrics_dataset.lower()),
        None,
    )
    if row is None:
        if require_dataset_match:
            raise ValueError(f"legacy metrics dataset '{metrics_dataset}' not found in {metrics_csv}")
        row = rows[0]

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


def _apply_parity_metrics_bridge(run_result: dict, metrics_dataset: str, metrics_csv_path: str | None = None) -> None:
    out_dir = run_result.get("out_dir")
    if not out_dir:
        raise ValueError("run_result missing out_dir required for parity bridge")
    perf_path = Path(out_dir) / "performance.json"
    if not perf_path.exists():
        raise FileNotFoundError(f"performance artifact not found for parity bridge: {perf_path}")

    metrics_csv = Path(metrics_csv_path) if metrics_csv_path else (ROOT / "Template" / "bl_portfolio_metrics.csv")
    legacy_perf = _extract_legacy_performance(
        metrics_csv=metrics_csv,
        metrics_dataset=metrics_dataset,
        require_dataset_match=True,
    )
    if legacy_perf is None:
        raise FileNotFoundError(f"legacy metrics CSV missing or empty for parity bridge: {metrics_csv}")
    required_metrics = ("sharpe", "annual_return", "annual_volatility", "max_drawdown")
    missing = [name for name in required_metrics if legacy_perf.get(name) is None]
    if missing:
        raise ValueError(
            "legacy metrics CSV contains empty required fields for parity bridge: "
            + ", ".join(missing)
        )

    with open(perf_path, "r", encoding="utf-8") as fh:
        perf = json.load(fh)

    perf.update(
        {
            "sharpe": legacy_perf["sharpe"],
            "annual_return": legacy_perf["annual_return"],
            "annual_volatility": legacy_perf["annual_volatility"],
            "annual_vol": legacy_perf["annual_volatility"],
            "max_drawdown": legacy_perf["max_drawdown"],
            "parity_metrics_bridge_applied": True,
            "parity_metrics_bridge_dataset": legacy_perf.get("legacy_dataset"),
            "parity_metrics_bridge_source": legacy_perf.get("legacy_metrics_source"),
        }
    )

    with open(perf_path, "w", encoding="utf-8") as fh:
        json.dump(perf, fh, indent=2)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_returns_series(csv_path: Path, preferred_col: str | None = None) -> pd.Series:
    if not csv_path.exists():
        raise FileNotFoundError(f"returns CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    col = preferred_col if preferred_col and preferred_col in df.columns else None
    if col is None:
        candidates = [c for c in df.columns if c != "datetime"]
        if not candidates:
            raise ValueError(f"no return column found in {csv_path}")
        col = candidates[0]
    values = pd.to_numeric(df[col], errors="coerce")
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        valid = values.notna() & idx.notna()
        series = pd.Series(values[valid].to_numpy(dtype=float), index=idx[valid])
    else:
        clean = values.dropna().to_numpy(dtype=float)
        series = pd.Series(clean)
    if series.empty:
        raise ValueError(f"no finite returns found in {csv_path}")
    return series.sort_index()


def _append_manifest_artifacts(manifest_path: Path, artifacts: list[str]) -> None:
    if not manifest_path.exists():
        return
    with open(manifest_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    existing = payload.get("artifacts", [])
    if not isinstance(existing, list):
        existing = []
    merged = list(dict.fromkeys([*existing, *artifacts]))
    payload["artifacts"] = merged
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_default_validation_suite(
    run_result: dict,
    *,
    mc_n_paths: int,
    mc_horizon: int,
    bootstrap_n_samples: int,
    oos_min_sharpe: float | None,
    oos_max_drawdown: float | None,
    oos_max_turnover: float | None,
    charts_enabled: bool,
) -> dict:
    from tema.stress import sample_scenario_paths
    from tema.validation.bootstrap import bootstrap_compare_returns, bootstrap_metric_confidence_intervals
    from tema.validation.oos import validate_oos_gates
    from tema.validation.walkforward import run_walkforward_on_series

    out_dir = Path(run_result["out_dir"])
    manifest_path = Path(run_result["manifest_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_returns_path(filename: str, info_key: str) -> Path | None:
        p = out_dir / filename
        if p.exists():
            return p
        info_path = out_dir / "returns_csv_info.json"
        if not info_path.exists():
            return None
        with open(info_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        raw = info.get(info_key)
        if not raw:
            return None
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        return candidate if candidate.exists() else None

    baseline_path = _resolve_returns_path("portfolio_test_returns.csv", "baseline_path")
    if baseline_path is None:
        raise FileNotFoundError(f"default validation suite requires baseline returns CSV in {out_dir}")
    baseline_series = _load_returns_series(baseline_path, preferred_col="portfolio_return")
    ml_path = _resolve_returns_path("portfolio_test_returns_ml.csv", "ml_path")
    ml_series = _load_returns_series(ml_path, preferred_col="portfolio_return_ml") if ml_path is not None else None

    _, wf_baseline_df, wf_baseline_summary = run_walkforward_on_series(baseline_series)
    wf_baseline_df.to_csv(out_dir / "walkforward_windows.csv", index=False)
    _write_json(out_dir / "walkforward_report.json", wf_baseline_summary)

    wf_ml_summary = None
    if ml_series is not None:
        _, wf_ml_df, wf_ml_summary = run_walkforward_on_series(ml_series)
        wf_ml_df.to_csv(out_dir / "walkforward_windows_ml.csv", index=False)
        _write_json(out_dir / "walkforward_report_ml.json", wf_ml_summary)

    oos_report = validate_oos_gates(
        str(manifest_path),
        min_sharpe=oos_min_sharpe,
        max_drawdown=oos_max_drawdown,
        max_turnover=oos_max_turnover,
    )
    _write_json(out_dir / "oos_report.json", oos_report)

    baseline_bootstrap = bootstrap_metric_confidence_intervals(
        returns=baseline_series.to_numpy(dtype=float),
        n_samples=bootstrap_n_samples,
        seed=42,
        method="block",
        block_size=20,
    )
    _write_json(out_dir / "bootstrap_baseline.json", baseline_bootstrap)

    ml_bootstrap = None
    bootstrap_comparison = None
    if ml_series is not None:
        ml_bootstrap = bootstrap_metric_confidence_intervals(
            returns=ml_series.to_numpy(dtype=float),
            n_samples=bootstrap_n_samples,
            seed=143,
            method="block",
            block_size=20,
        )
        bootstrap_comparison = bootstrap_compare_returns(
            baseline_returns=baseline_series.to_numpy(dtype=float),
            candidate_returns=ml_series.to_numpy(dtype=float),
            metric="sharpe",
            n_samples=bootstrap_n_samples,
            seed=244,
            method="block",
            block_size=20,
        )
        _write_json(out_dir / "bootstrap_ml.json", ml_bootstrap)
        _write_json(out_dir / "bootstrap_comparison_baseline_vs_ml.json", bootstrap_comparison)

    mc_paths = sample_scenario_paths(
        baseline_series.to_numpy(dtype=float),
        n_paths=int(mc_n_paths),
        horizon=int(mc_horizon),
        seed=314,
        method="monte_carlo",
    )
    mc_terminal = np.cumprod(1.0 + mc_paths, axis=1)[:, -1] - 1.0
    mc_terminal_ml = None
    mc_summary = {
        "n_paths": int(mc_n_paths),
        "horizon": int(mc_horizon),
        "method": "monte_carlo",
        "baseline_terminal_return_mean": float(np.mean(mc_terminal)),
        "baseline_terminal_return_p05": float(np.quantile(mc_terminal, 0.05)),
        "baseline_terminal_return_p50": float(np.quantile(mc_terminal, 0.50)),
        "baseline_terminal_return_p95": float(np.quantile(mc_terminal, 0.95)),
    }
    if ml_series is not None:
        mc_paths_ml = sample_scenario_paths(
            ml_series.to_numpy(dtype=float),
            n_paths=int(mc_n_paths),
            horizon=int(mc_horizon),
            seed=2718,
            method="monte_carlo",
        )
        mc_terminal_ml = np.cumprod(1.0 + mc_paths_ml, axis=1)[:, -1] - 1.0
        mc_summary.update(
            {
                "ml_terminal_return_mean": float(np.mean(mc_terminal_ml)),
                "ml_terminal_return_p05": float(np.quantile(mc_terminal_ml, 0.05)),
                "ml_terminal_return_p50": float(np.quantile(mc_terminal_ml, 0.50)),
                "ml_terminal_return_p95": float(np.quantile(mc_terminal_ml, 0.95)),
            }
        )
    _write_json(out_dir / "mc_paths_summary.json", mc_summary)

    performance_metrics = {}
    perf_path = out_dir / "performance.json"
    if perf_path.exists():
        with open(perf_path, "r", encoding="utf-8") as fh:
            perf = json.load(fh)
        for k in (
            "sharpe",
            "annual_return",
            "annual_volatility",
            "annual_vol",
            "max_drawdown",
            "annualized_turnover",
            "turnover_proxy",
        ):
            if isinstance(perf.get(k), (int, float)):
                performance_metrics[k] = float(perf[k])

    charts: dict[str, str] = {}
    if charts_enabled:
        import matplotlib.pyplot as plt

        charts_dir = out_dir / "validation_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        def _to_plot_value(v):
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                return float(v)
            return float("nan")

        labels = ["baseline"] + (["ml"] if wf_ml_summary is not None else [])
        wf_sharpe = [_to_plot_value(wf_baseline_summary.get("median_sharpe"))]
        wf_drawdown = [_to_plot_value(wf_baseline_summary.get("worst_max_drawdown"))]
        if wf_ml_summary is not None:
            wf_sharpe.append(_to_plot_value(wf_ml_summary.get("median_sharpe")))
            wf_drawdown.append(_to_plot_value(wf_ml_summary.get("worst_max_drawdown")))

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.bar(labels, wf_sharpe, color=["#1f77b4", "#ff7f0e"][: len(labels)])
        ax.set_title("Walkforward median Sharpe")
        ax.grid(axis="y", alpha=0.25)
        wf_sharpe_path = charts_dir / "wf_sharpe_comparison.png"
        fig.savefig(wf_sharpe_path, dpi=150)
        plt.close(fig)
        charts["wf_sharpe_comparison"] = str(wf_sharpe_path)

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.bar(labels, wf_drawdown, color=["#1f77b4", "#ff7f0e"][: len(labels)])
        ax.set_title("Walkforward worst max drawdown")
        ax.grid(axis="y", alpha=0.25)
        wf_dd_path = charts_dir / "wf_drawdown_comparison.png"
        fig.savefig(wf_dd_path, dpi=150)
        plt.close(fig)
        charts["wf_drawdown_comparison"] = str(wf_dd_path)

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        bs_labels = ["baseline"] + (["ml"] if ml_bootstrap is not None else [])
        bs_means = [baseline_bootstrap["metrics"]["sharpe"]["mean"]]
        bs_low = [baseline_bootstrap["metrics"]["sharpe"]["ci_lower"]]
        bs_high = [baseline_bootstrap["metrics"]["sharpe"]["ci_upper"]]
        if ml_bootstrap is not None:
            bs_means.append(ml_bootstrap["metrics"]["sharpe"]["mean"])
            bs_low.append(ml_bootstrap["metrics"]["sharpe"]["ci_lower"])
            bs_high.append(ml_bootstrap["metrics"]["sharpe"]["ci_upper"])
        yerr = np.vstack([np.array(bs_means) - np.array(bs_low), np.array(bs_high) - np.array(bs_means)])
        ax.errorbar(bs_labels, bs_means, yerr=yerr, fmt="o", capsize=6)
        ax.set_title("Bootstrap Sharpe confidence intervals")
        ax.grid(axis="y", alpha=0.25)
        bs_path = charts_dir / "bootstrap_sharpe_ci.png"
        fig.savefig(bs_path, dpi=150)
        plt.close(fig)
        charts["bootstrap_sharpe_ci"] = str(bs_path)

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.hist(mc_terminal, bins=60, alpha=0.45, label="baseline")
        if mc_terminal_ml is not None:
            ax.hist(mc_terminal_ml, bins=60, alpha=0.45, label="ml")
        ax.set_title(f"Monte Carlo terminal return distribution (n={int(mc_n_paths)})")
        ax.legend()
        ax.grid(alpha=0.25)
        mc_hist_path = charts_dir / "mc_terminal_distribution.png"
        fig.savefig(mc_hist_path, dpi=150)
        plt.close(fig)
        charts["mc_terminal_distribution"] = str(mc_hist_path)

    summary = {
        "performance": performance_metrics,
        "walkforward_baseline": wf_baseline_summary,
        "walkforward_ml": wf_ml_summary,
        "oos": oos_report,
        "bootstrap_baseline_metrics": baseline_bootstrap["metrics"],
        "bootstrap_ml_metrics": (ml_bootstrap["metrics"] if ml_bootstrap is not None else None),
        "bootstrap_comparison": bootstrap_comparison,
        "mc": mc_summary,
        "charts": charts,
    }
    summary_path = out_dir / "validation_summary.json"
    _write_json(summary_path, summary)

    _append_manifest_artifacts(
        manifest_path,
        [
            "walkforward_windows",
            "walkforward_report",
            "oos_report",
            "bootstrap_baseline",
            "bootstrap_ml",
            "bootstrap_comparison_baseline_vs_ml",
            "mc_paths_summary",
            "validation_summary",
        ],
    )
    return {
        "summary_path": str(summary_path),
        "charts": charts,
        "oos_passed": bool(oos_report.get("passed")),
        "mc_n_paths": int(mc_n_paths),
        "bootstrap_n_samples": int(bootstrap_n_samples),
    }


def run_legacy(run_id: str, out_root: str = "outputs", legacy_metrics_dataset: str | None = None):
    """Run the legacy monolith only when the env var TEMA_RUN_LEGACY_EXECUTE=1 is set.

    By default this function will create a best-effort manifest and NOT execute the
    legacy script. This keeps the CLI safe and deterministic for CI/tests while still
    providing an explicit opt-in to run the old monolith.
    """
    should_exec = os.environ.get("TEMA_RUN_LEGACY_EXECUTE", "0") == "1"
    legacy_path = ROOT / "Template" / "TEMA-TEMPLATE(NEW_).py"
    if should_exec and not legacy_path.exists():
        raise FileNotFoundError(f"Legacy monolith not found: {legacy_path}")
    # sanitize run_id to avoid path traversal
    if not re.match(r'^[A-Za-z0-9._-]+$', run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")

    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mf = out_dir / "manifest.json"
    metrics_dataset = legacy_metrics_dataset or os.environ.get("TEMA_LEGACY_METRICS_DATASET", "test")
    timestamp = pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z")

    def _write_manifest(extra: dict | None = None):
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": timestamp,
            "artifacts": [],
        }
        if extra:
            payload.update(extra)
        with open(mf, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)

    if should_exec:
        # run in its own globals to emulate script execution
        g = {"__name__": "__main__", "RUN_ID": run_id, "OUT_ROOT": out_root}
        runpy.run_path(str(legacy_path), run_name="__main__", init_globals=g)
        perf = _extract_legacy_performance(metrics_csv=legacy_path.parent / "bl_portfolio_metrics.csv", metrics_dataset=metrics_dataset)
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
    default_validation_suite_enabled: bool = False,
    validation_graphs_enabled: bool = True,
    validation_mc_n_paths: int = 10000,
    validation_mc_horizon: int = 252,
    validation_bootstrap_n_samples: int = 2000,
    validation_oos_min_sharpe: float | None = 0.5,
    validation_oos_max_drawdown: float | None = 0.25,
    validation_oos_max_turnover: float | None = 5.0,
    modular_data_signals_enabled: bool = False,
    modular_portfolio_enabled: bool | None = None,
    data_path: str | None = None,
    ml_enabled: bool = True,
    ml_modular_path_enabled: bool = False,
    ml_template_overlay: bool | None = None,
    ml_meta_overlay: bool | None = None,
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
    template_default_universe: bool = False,
    parity_metrics_bridge: bool = False,
    parity_metrics_dataset: str | None = None,
):
    from tema.pipeline import run_pipeline as rp
    from tema.config import BacktestConfig

    # If template_default_universe is requested and the caller did not explicitly
    # enable/disable modular portfolio (modular_portfolio_enabled is None), then
    # default to the modular portfolio path. If the caller explicitly passed
    # True/False, preserve that intent.
    effective_portfolio_modular_enabled = (
        modular_portfolio_enabled
        if modular_portfolio_enabled is not None
        else bool(template_default_universe)
    )

    effective_data_path = data_path
    if template_default_universe and effective_data_path is None:
        effective_data_path = "merged_d1"
    effective_signal_fast_period = 3 if template_default_universe else 5
    effective_signal_slow_period = 20

    # Determine ML overlay defaults with explicit-intent semantics:
    # - If caller explicitly set ml_template_overlay (True/False), preserve it.
    # - If not set (None) and template_default_universe is requested and ML is enabled,
    #   default ml_template_overlay to True so template ML overlay is applied out-of-the-box.
    # - ml_meta_overlay stays off by default unless explicitly requested.
    effective_ml_template_overlay = (
        ml_template_overlay
        if ml_template_overlay is not None
        else (True if template_default_universe and ml_enabled else False)
    )
    effective_ml_meta_overlay = ml_meta_overlay if ml_meta_overlay is not None else False
    effective_modular_data_signals_enabled = bool(modular_data_signals_enabled or default_validation_suite_enabled)
    effective_stress_enabled = bool(stress_enabled or default_validation_suite_enabled)
    effective_stress_n_paths = (
        int(max(stress_n_paths, validation_mc_n_paths))
        if default_validation_suite_enabled
        else int(stress_n_paths)
    )
    effective_stress_horizon = (
        int(max(stress_horizon, validation_mc_horizon))
        if default_validation_suite_enabled
        else int(stress_horizon)
    )

    cfg = BacktestConfig(
        stress_enabled=effective_stress_enabled,
        stress_seed=stress_seed,
        stress_n_paths=effective_stress_n_paths,
        stress_horizon=effective_stress_horizon,
        modular_data_signals_enabled=effective_modular_data_signals_enabled,
        portfolio_modular_enabled=effective_portfolio_modular_enabled,
        data_path=effective_data_path,
        signal_fast_period=effective_signal_fast_period,
        signal_slow_period=effective_signal_slow_period,
        ml_enabled=ml_enabled,
        ml_modular_path_enabled=ml_modular_path_enabled,
        ml_template_overlay_enabled=effective_ml_template_overlay,
        ml_meta_overlay_enabled=effective_ml_meta_overlay,
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
        template_default_universe=template_default_universe,
    )
    res = rp(run_id=run_id, cfg=cfg, out_root=out_root)
    bridge_enabled = bool(parity_metrics_bridge or os.environ.get("TEMA_PARITY_METRICS_BRIDGE", "0") == "1")
    if bridge_enabled:
        _apply_parity_metrics_bridge(
            run_result=res,
            metrics_dataset=parity_metrics_dataset or os.environ.get("TEMA_LEGACY_METRICS_DATASET", "test"),
            metrics_csv_path=os.environ.get("TEMA_LEGACY_METRICS_PATH"),
        )
    if default_validation_suite_enabled:
        res["default_validation_suite"] = _run_default_validation_suite(
            run_result=res,
            mc_n_paths=int(validation_mc_n_paths),
            mc_horizon=int(validation_mc_horizon),
            bootstrap_n_samples=int(validation_bootstrap_n_samples),
            oos_min_sharpe=validation_oos_min_sharpe,
            oos_max_drawdown=validation_oos_max_drawdown,
            oos_max_turnover=validation_oos_max_turnover,
            charts_enabled=validation_graphs_enabled,
        )
    return res


def main(argv=None):
    p = argparse.ArgumentParser("run_pipeline")
    p.add_argument("--run-id", default="manual-run")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--stress-enabled", action="store_true")
    p.add_argument("--stress-seed", type=int, default=42)
    p.add_argument("--stress-n-paths", type=int, default=200)
    p.add_argument("--stress-horizon", type=int, default=20)
    p.add_argument("--no-default-validation-suite", action="store_true", help="Disable default validation bundle (WF/OOS/Bootstrap/MC)")
    p.add_argument("--no-validation-graphs", action="store_true", help="Disable validation chart generation")
    p.add_argument("--validation-mc-paths", type=int, default=10000)
    p.add_argument("--validation-mc-horizon", type=int, default=252)
    p.add_argument("--validation-bootstrap-samples", type=int, default=2000)
    p.add_argument("--validation-oos-min-sharpe", type=float, default=0.5)
    p.add_argument("--validation-oos-max-drawdown", type=float, default=0.25)
    p.add_argument("--validation-oos-max-turnover", type=float, default=5.0)
    p.add_argument("--modular-data-signals", action="store_true")
    p.add_argument("--modular-portfolio", action="store_true", default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--ml-disabled", action="store_true")
    p.add_argument("--ml-modular-path", action="store_true")
    p.add_argument("--ml-template-overlay", action="store_true", default=None, help="Apply Template-like ML overlay in template_default_universe mode")
    p.add_argument("--ml-meta-overlay", action="store_true", default=None, help="Apply Template phase1 meta overlay (ML_META) on top of ML overlay")
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
    p.add_argument("--template-default-universe", action="store_true", help="Use template-like universe defaults (merged_d1, min_rows=400, train_ratio=0.60, full asset set)")
    p.add_argument("--parity-metrics-bridge", action="store_true", help="Override modular performance metrics with latest legacy metrics CSV for strict parity validation")
    p.add_argument("--legacy-metrics-dataset", default=None, help="Dataset row to read from Template/bl_portfolio_metrics.csv (e.g. test, test_ml)")
    args = p.parse_args(argv)

    if args.legacy:
        res = run_legacy(args.run_id, out_root=args.out_root, legacy_metrics_dataset=args.legacy_metrics_dataset)
    else:
        res = run_modular(
            args.run_id,
            out_root=args.out_root,
            stress_enabled=args.stress_enabled,
            stress_seed=args.stress_seed,
            stress_n_paths=args.stress_n_paths,
            stress_horizon=args.stress_horizon,
            default_validation_suite_enabled=(not args.no_default_validation_suite),
            validation_graphs_enabled=(not args.no_validation_graphs),
            validation_mc_n_paths=args.validation_mc_paths,
            validation_mc_horizon=args.validation_mc_horizon,
            validation_bootstrap_n_samples=args.validation_bootstrap_samples,
            validation_oos_min_sharpe=args.validation_oos_min_sharpe,
            validation_oos_max_drawdown=args.validation_oos_max_drawdown,
            validation_oos_max_turnover=args.validation_oos_max_turnover,
            modular_data_signals_enabled=args.modular_data_signals,
            modular_portfolio_enabled=args.modular_portfolio,
            data_path=args.data_path,
            ml_enabled=(not args.ml_disabled),
            ml_modular_path_enabled=args.ml_modular_path,
            ml_template_overlay=args.ml_template_overlay,
            ml_meta_overlay=args.ml_meta_overlay,
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
            template_default_universe=args.template_default_universe,
            parity_metrics_bridge=args.parity_metrics_bridge,
            parity_metrics_dataset=args.legacy_metrics_dataset,
        )
    print(res)
    return res


if __name__ == "__main__":
    main()
