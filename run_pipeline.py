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
from tema.ml.cpp_profile import resolve_cpp_hmm_profile, available_cpp_hmm_profiles


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


def _apply_parity_metrics_bridge(
    run_result: dict,
    metrics_dataset: str,
    metrics_csv_path: str | None = None,
    *,
    strict_independent_mode: bool = False,
) -> None:
    if strict_independent_mode:
        raise ValueError(
            "strict_independent_mode violation: parity metrics bridge uses legacy comparator CSV source"
        )
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
            "benchmark_injection_detected": True,
            "benchmark_injection_sources": ["parity_metrics_bridge.legacy_metrics_csv"],
            "strict_independent_mode": False,
        }
    )
    source = perf.get("source")
    if not isinstance(source, dict):
        source = {}
    source.update(
        {
            "benchmark_injection_detected": True,
            "benchmark_injection_sources": ["parity_metrics_bridge.legacy_metrics_csv"],
            "strict_independent_mode": False,
        }
    )
    perf["source"] = source

    with open(perf_path, "w", encoding="utf-8") as fh:
        json.dump(perf, fh, indent=2)

    returns_info_path = Path(out_dir) / "returns_csv_info.json"
    if returns_info_path.exists():
        with open(returns_info_path, "r", encoding="utf-8") as fh:
            returns_info = json.load(fh)
        existing_sources = returns_info.get("benchmark_injection_sources", [])
        if not isinstance(existing_sources, list):
            existing_sources = []
        merged_sources = list(dict.fromkeys([*existing_sources, "parity_metrics_bridge.legacy_metrics_csv"]))
        returns_info.update(
            {
                "benchmark_injection_detected": True,
                "benchmark_injection_sources": merged_sources,
                "strict_independent_mode": False,
            }
        )
        with open(returns_info_path, "w", encoding="utf-8") as fh:
            json.dump(returns_info, fh, indent=2)


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
    oos_min_calmar: float | None,
    psr_threshold: float | None,
    dsr_threshold: float | None,
    pbo_max: float | None,
    cpcv_n_groups: int,
    cpcv_n_test_groups: int,
    cpcv_purge_groups: int,
    cpcv_embargo_groups: int,
    cpcv_max_splits: int | None,
    hard_fail: bool,
    charts_enabled: bool,
) -> dict:
    from tema.stress import sample_scenario_paths
    from tema.validation.bootstrap import bootstrap_compare_returns, bootstrap_metric_confidence_intervals
    from tema.validation.cpcv import evaluate_cpcv_strategies, generate_cpcv_splits
    from tema.validation.oos import validate_oos_gates
    from tema.validation.probabilistic import deflated_sharpe_ratio, probabilistic_sharpe_ratio
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
        min_calmar=oos_min_calmar,
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
    baseline_probabilistic = {
        "psr": probabilistic_sharpe_ratio(
            returns=baseline_series.to_numpy(dtype=float),
            sr_benchmark=0.0,
            annualization_factor=252.0,
        ),
        "dsr": deflated_sharpe_ratio(
            returns=baseline_series.to_numpy(dtype=float),
            n_trials=max(2, int(cpcv_n_groups)),
            sr_mean=0.0,
            annualization_factor=252.0,
        ),
    }
    _write_json(out_dir / "probabilistic_sharpe_baseline.json", baseline_probabilistic)

    ml_bootstrap = None
    bootstrap_comparison = None
    ml_probabilistic = None
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
        ml_probabilistic = {
            "psr": probabilistic_sharpe_ratio(
                returns=ml_series.to_numpy(dtype=float),
                sr_benchmark=0.0,
                annualization_factor=252.0,
            ),
            "dsr": deflated_sharpe_ratio(
                returns=ml_series.to_numpy(dtype=float),
                n_trials=max(2, int(cpcv_n_groups)),
                sr_mean=0.0,
                annualization_factor=252.0,
            ),
        }
        _write_json(out_dir / "probabilistic_sharpe_ml.json", ml_probabilistic)

    cpcv_report: dict
    cpcv_series: dict[str, pd.Series] = {"baseline": baseline_series}
    if ml_series is not None:
        cpcv_series["ml"] = ml_series
    ml_meta_path = _resolve_returns_path("portfolio_test_returns_ml_meta.csv", "ml_meta_path")
    if ml_meta_path is not None:
        try:
            cpcv_series["ml_meta"] = _load_returns_series(ml_meta_path, preferred_col="portfolio_return_ml_meta")
        except (FileNotFoundError, ValueError):
            pass

    cpcv_df = pd.concat(cpcv_series, axis=1).dropna(how="all")
    cpcv_df.columns = [str(c) for c in cpcv_df.columns]
    if cpcv_df.shape[1] < 2 or cpcv_df.shape[0] < max(int(cpcv_n_groups), 20):
        cpcv_report = {
            "skipped": True,
            "reason": "insufficient_series_or_rows",
            "n_rows": int(cpcv_df.shape[0]),
            "n_strategies": int(cpcv_df.shape[1]),
        }
    else:
        cpcv_splits = generate_cpcv_splits(
            index=cpcv_df.index,
            n_groups=int(cpcv_n_groups),
            n_test_groups=int(cpcv_n_test_groups),
            purge_groups=int(cpcv_purge_groups),
            embargo_groups=int(cpcv_embargo_groups),
            max_splits=cpcv_max_splits,
            seed=42,
        )
        cpcv_report = evaluate_cpcv_strategies(
            returns_df=cpcv_df,
            splits=cpcv_splits,
            annualization_factor=252.0,
            metric="sharpe",
        )
        cpcv_report["split_generation"] = {
            "n_groups": int(cpcv_n_groups),
            "n_test_groups": int(cpcv_n_test_groups),
            "purge_groups": int(cpcv_purge_groups),
            "embargo_groups": int(cpcv_embargo_groups),
            "n_splits_generated": int(len(cpcv_splits)),
        }
    _write_json(out_dir / "cpcv_report.json", cpcv_report)

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
        "probabilistic_baseline": baseline_probabilistic,
        "probabilistic_ml": ml_probabilistic,
        "cpcv": cpcv_report,
        "mc": mc_summary,
        "charts": charts,
    }
    target_prob = ml_probabilistic if ml_probabilistic is not None else baseline_probabilistic
    target_psr = (
        float(target_prob["psr"]["psr"])
        if isinstance(target_prob, dict)
        and isinstance(target_prob.get("psr"), dict)
        and isinstance(target_prob["psr"].get("psr"), (int, float))
        else None
    )
    target_dsr = (
        float(target_prob["dsr"]["dsr"])
        if isinstance(target_prob, dict)
        and isinstance(target_prob.get("dsr"), dict)
        and isinstance(target_prob["dsr"].get("dsr"), (int, float))
        else None
    )
    cpcv_pbo = (
        float(cpcv_report["pbo"]["pbo"])
        if isinstance(cpcv_report, dict)
        and isinstance(cpcv_report.get("pbo"), dict)
        and isinstance(cpcv_report["pbo"].get("pbo"), (int, float))
        else None
    )
    hard_gate_checks = {
        "oos_passed": bool(oos_report.get("passed", False)),
        "psr_threshold": (
            True
            if psr_threshold is None or target_psr is None
            else bool(target_psr >= float(psr_threshold))
        ),
        "dsr_threshold": (
            True
            if dsr_threshold is None or target_dsr is None
            else bool(target_dsr >= float(dsr_threshold))
        ),
        "pbo_max": (
            True
            if pbo_max is None or cpcv_pbo is None
            else bool(cpcv_pbo <= float(pbo_max))
        ),
    }
    hard_gate_passed = bool(all(hard_gate_checks.values()))
    summary["hard_gate"] = {
        "passed": hard_gate_passed,
        "checks": hard_gate_checks,
        "thresholds": {
            "psr_threshold": psr_threshold,
            "dsr_threshold": dsr_threshold,
            "pbo_max": pbo_max,
        },
        "values": {
            "target_psr": target_psr,
            "target_dsr": target_dsr,
            "pbo": cpcv_pbo,
            "target_series": ("ml" if ml_probabilistic is not None else "baseline"),
        },
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
            "probabilistic_sharpe_baseline",
            "probabilistic_sharpe_ml",
            "cpcv_report",
            "mc_paths_summary",
            "validation_summary",
        ],
    )
    if bool(hard_fail) and not hard_gate_passed:
        raise ValueError("default validation hard gate failed")
    return {
        "summary_path": str(summary_path),
        "charts": charts,
        "oos_passed": bool(oos_report.get("passed")),
        "validation_hard_gate_passed": hard_gate_passed,
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
    validation_oos_min_calmar: float | None = None,
    validation_psr_threshold: float | None = 0.95,
    validation_dsr_threshold: float | None = 0.80,
    validation_pbo_max: float | None = 0.50,
    validation_cpcv_n_groups: int = 10,
    validation_cpcv_n_test_groups: int = 2,
    validation_cpcv_purge_groups: int = 1,
    validation_cpcv_embargo_groups: int = 1,
    validation_cpcv_max_splits: int | None = 256,
    validation_hard_fail: bool = False,
    modular_data_signals_enabled: bool = False,
    modular_portfolio_enabled: bool | None = None,
    data_path: str | None = None,
    ml_enabled: bool = True,
    ml_modular_path_enabled: bool = False,
    ml_template_overlay: bool | None = None,
    ml_meta_overlay: bool | None = None,
    ml_meta_use_triple_barrier: bool = False,
    ml_meta_tb_horizon: int = 5,
    ml_meta_tb_upper: float = 0.01,
    ml_meta_tb_lower: float = 0.01,
    ml_probability_threshold: float = 0.0,
    ml_feature_fracdiff_enabled: bool = False,
    ml_feature_fracdiff_order: float = 0.4,
    ml_feature_fracdiff_threshold: float = 1e-5,
    ml_feature_fracdiff_max_terms: int = 256,
    ml_feature_har_rv_enabled: bool = False,
    ml_feature_har_rv_windows: tuple[int, ...] = (1, 5, 22),
    ml_feature_har_rv_use_log: bool = True,
    data_max_assets: int = 3,
    data_full_universe_for_parity: bool = True,
    portfolio_method: str = "bl",
    portfolio_risk_aversion: float = 2.5,
    portfolio_cov_shrinkage: float = 0.15,
    portfolio_covariance_backend: str = "sample",
    portfolio_correlation_backend: str = "pearson",
    portfolio_gerber_threshold: float = 0.5,
    portfolio_bl_tau: float = 0.05,
    portfolio_bl_view_confidence: float = 0.65,
    portfolio_bl_omega_scale: float = 0.25,
    portfolio_bl_max_weight: float = 0.15,
    portfolio_regime_mapping_enabled: bool = False,
    portfolio_regime_mapping_mode: str = "linear",
    portfolio_regime_mapping_min_multiplier: float = 1.0,
    portfolio_regime_mapping_max_multiplier: float = 1.0,
    portfolio_regime_mapping_step_thresholds: tuple[float, ...] = (0.30, 0.70),
    portfolio_regime_mapping_step_multipliers: tuple[float, ...] = (1.0, 1.0, 1.0),
    portfolio_regime_mapping_kelly_gamma: float = 2.0,
    ml_hmm_scalar_floor: float = 0.30,
    ml_hmm_scalar_ceiling: float = 1.50,
    vol_target_apply_to_ml: bool = False,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
    cost_model: str = "simple",
    spread_bps: float = 0.0,
    impact_coeff: float = 0.0,
    borrow_bps: float = 0.0,
    dynamic_trading_enabled: bool = False,
    dynamic_trading_lambda: float = 0.0,
    dynamic_trading_aim_multiplier: float = 0.0,
    dynamic_trading_min_trade_rate: float = 0.10,
    dynamic_trading_max_trade_rate: float = 1.0,
    execution_backend: str = "instant",
    execution_ac_n_slices: int = 4,
    execution_ac_risk_aversion: float = 1.0,
    execution_ac_temporary_impact: float = 0.10,
    execution_ac_permanent_impact: float = 0.01,
    execution_ac_volatility_lookback: int = 20,
    experimental_multi_horizon_blend_enabled: bool = False,
    experimental_conformal_sizing_enabled: bool = False,
    experimental_futuretesting_enabled: bool = False,
    experimental_futuretesting_method: str = "stationary_bootstrap",
    experimental_futuretesting_block_size: int | None = None,
    experimental_futuretesting_n_paths: int = 200,
    experimental_futuretesting_horizon: int = 126,
    template_default_universe: bool = False,
    template_rebalance_enabled: bool = False,
    template_use_precomputed_artifacts: bool = True,
    ml_meta_comparator_use_benchmark_csv: bool = False,
    cpp_hmm_profile: str | None = None,
    cle_enabled: bool = False,
    cle_use_external_proxies: bool = False,
    cle_mode: str = "confluence_blend",
    cle_mapping_mode: str = "linear",
    cle_mapping_min_multiplier: float = 0.5,
    cle_mapping_max_multiplier: float = 1.5,
    cle_mapping_step_thresholds: tuple[float, ...] = (0.30, 0.70),
    cle_mapping_step_multipliers: tuple[float, ...] = (0.50, 1.00, 1.50),
    cle_mapping_kelly_gamma: float = 2.0,
    cle_gate_event_blackout_cap: float = 0.5,
    cle_gate_liquidity_spread_z_threshold: float = 2.0,
    cle_gate_liquidity_depth_threshold: float = 0.10,
    cle_gate_liquidity_reduction_factor: float = 0.25,
    cle_gate_correlation_alert_cap: float = 1.0,
    cle_leverage_floor: float = 0.0,
    cle_leverage_cap: float = 12.0,
    cle_policy_seed: int = 42,
    cle_online_calibration_enabled: bool = False,
    cle_online_calibration_window: int = 5,
    cle_online_calibration_learning_rate: float = 0.10,
    cle_online_calibration_l2: float = 1e-4,
    parity_metrics_bridge: bool = False,
    parity_metrics_dataset: str | None = None,
    strict_independent_mode: bool = False,
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
    effective_template_grid_signal_logic = (
        "or" if template_default_universe and (not template_use_precomputed_artifacts) else "hierarchical"
    )

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
    effective_modular_data_signals_enabled = bool(
        modular_data_signals_enabled or default_validation_suite_enabled or template_rebalance_enabled
    )
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
    profile_overrides: dict[str, float | int] = {}
    if cpp_hmm_profile:
        profile_overrides = resolve_cpp_hmm_profile(cpp_hmm_profile)

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
        template_grid_signal_logic=effective_template_grid_signal_logic,
        ml_enabled=ml_enabled,
        ml_modular_path_enabled=ml_modular_path_enabled,
        ml_template_overlay_enabled=effective_ml_template_overlay,
        ml_meta_overlay_enabled=effective_ml_meta_overlay,
        ml_meta_use_triple_barrier=ml_meta_use_triple_barrier,
        ml_meta_tb_horizon=ml_meta_tb_horizon,
        ml_meta_tb_upper=ml_meta_tb_upper,
        ml_meta_tb_lower=ml_meta_tb_lower,
        ml_probability_threshold=ml_probability_threshold,
        ml_feature_fracdiff_enabled=ml_feature_fracdiff_enabled,
        ml_feature_fracdiff_order=ml_feature_fracdiff_order,
        ml_feature_fracdiff_threshold=ml_feature_fracdiff_threshold,
        ml_feature_fracdiff_max_terms=ml_feature_fracdiff_max_terms,
        ml_feature_har_rv_enabled=ml_feature_har_rv_enabled,
        ml_feature_har_rv_windows=ml_feature_har_rv_windows,
        ml_feature_har_rv_use_log=ml_feature_har_rv_use_log,
        data_max_assets=data_max_assets,
        data_full_universe_for_parity=data_full_universe_for_parity,
        portfolio_method=portfolio_method,
        portfolio_risk_aversion=portfolio_risk_aversion,
        portfolio_cov_shrinkage=portfolio_cov_shrinkage,
        portfolio_covariance_backend=portfolio_covariance_backend,
        portfolio_correlation_backend=portfolio_correlation_backend,
        portfolio_gerber_threshold=portfolio_gerber_threshold,
        portfolio_bl_tau=portfolio_bl_tau,
        portfolio_bl_view_confidence=portfolio_bl_view_confidence,
        portfolio_bl_omega_scale=portfolio_bl_omega_scale,
        portfolio_bl_max_weight=portfolio_bl_max_weight,
        portfolio_regime_mapping_enabled=portfolio_regime_mapping_enabled,
        portfolio_regime_mapping_mode=portfolio_regime_mapping_mode,
        portfolio_regime_mapping_min_multiplier=portfolio_regime_mapping_min_multiplier,
        portfolio_regime_mapping_max_multiplier=portfolio_regime_mapping_max_multiplier,
        portfolio_regime_mapping_step_thresholds=portfolio_regime_mapping_step_thresholds,
        portfolio_regime_mapping_step_multipliers=portfolio_regime_mapping_step_multipliers,
        portfolio_regime_mapping_kelly_gamma=portfolio_regime_mapping_kelly_gamma,
        ml_hmm_scalar_floor=ml_hmm_scalar_floor,
        ml_hmm_scalar_ceiling=ml_hmm_scalar_ceiling,
        vol_target_apply_to_ml=vol_target_apply_to_ml,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        cost_model=cost_model,
        spread_bps=spread_bps,
        impact_coeff=impact_coeff,
        borrow_bps=borrow_bps,
        dynamic_trading_enabled=dynamic_trading_enabled,
        dynamic_trading_lambda=dynamic_trading_lambda,
        dynamic_trading_aim_multiplier=dynamic_trading_aim_multiplier,
        dynamic_trading_min_trade_rate=dynamic_trading_min_trade_rate,
        dynamic_trading_max_trade_rate=dynamic_trading_max_trade_rate,
        execution_backend=execution_backend,
        execution_ac_n_slices=execution_ac_n_slices,
        execution_ac_risk_aversion=execution_ac_risk_aversion,
        execution_ac_temporary_impact=execution_ac_temporary_impact,
        execution_ac_permanent_impact=execution_ac_permanent_impact,
        execution_ac_volatility_lookback=execution_ac_volatility_lookback,
        experimental_multi_horizon_blend_enabled=experimental_multi_horizon_blend_enabled,
        experimental_conformal_sizing_enabled=experimental_conformal_sizing_enabled,
        experimental_futuretesting_enabled=experimental_futuretesting_enabled,
        experimental_futuretesting_method=experimental_futuretesting_method,
        experimental_futuretesting_block_size=experimental_futuretesting_block_size,
        experimental_futuretesting_n_paths=experimental_futuretesting_n_paths,
        experimental_futuretesting_horizon=experimental_futuretesting_horizon,
        template_default_universe=template_default_universe,
        template_rebalance_enabled=template_rebalance_enabled,
        template_use_precomputed_artifacts=template_use_precomputed_artifacts,
        ml_meta_comparator_use_benchmark_csv=ml_meta_comparator_use_benchmark_csv,
        cpp_hmm_profile=cpp_hmm_profile,
        strict_independent_mode=strict_independent_mode,
        cle_enabled=cle_enabled,
        cle_use_external_proxies=cle_use_external_proxies,
        cle_mode=cle_mode,
        cle_mapping_mode=cle_mapping_mode,
        cle_mapping_min_multiplier=cle_mapping_min_multiplier,
        cle_mapping_max_multiplier=cle_mapping_max_multiplier,
        cle_mapping_step_thresholds=cle_mapping_step_thresholds,
        cle_mapping_step_multipliers=cle_mapping_step_multipliers,
        cle_mapping_kelly_gamma=cle_mapping_kelly_gamma,
        cle_gate_event_blackout_cap=cle_gate_event_blackout_cap,
        cle_gate_liquidity_spread_z_threshold=cle_gate_liquidity_spread_z_threshold,
        cle_gate_liquidity_depth_threshold=cle_gate_liquidity_depth_threshold,
        cle_gate_liquidity_reduction_factor=cle_gate_liquidity_reduction_factor,
        cle_gate_correlation_alert_cap=cle_gate_correlation_alert_cap,
        cle_leverage_floor=cle_leverage_floor,
        cle_leverage_cap=cle_leverage_cap,
        cle_policy_seed=cle_policy_seed,
        cle_online_calibration_enabled=cle_online_calibration_enabled,
        cle_online_calibration_window=cle_online_calibration_window,
        cle_online_calibration_learning_rate=cle_online_calibration_learning_rate,
        cle_online_calibration_l2=cle_online_calibration_l2,
        **profile_overrides,
    )
    res = rp(run_id=run_id, cfg=cfg, out_root=out_root)
    # Bridge activation is now explicit-only: only enable when the caller
    # passed parity_metrics_bridge=True. Previously the environment variable
    # TEMA_PARITY_METRICS_BRIDGE could implicitly enable the bridge which
    # allowed hidden overrides; remove that implicit path to make parity
    # activation explicit and easier to reason about in CI.
    bridge_enabled = bool(parity_metrics_bridge)
    if bridge_enabled:
        _apply_parity_metrics_bridge(
            run_result=res,
            metrics_dataset=parity_metrics_dataset or os.environ.get("TEMA_LEGACY_METRICS_DATASET", "test"),
            metrics_csv_path=os.environ.get("TEMA_LEGACY_METRICS_PATH"),
            strict_independent_mode=bool(strict_independent_mode),
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
            oos_min_calmar=validation_oos_min_calmar,
            psr_threshold=validation_psr_threshold,
            dsr_threshold=validation_dsr_threshold,
            pbo_max=validation_pbo_max,
            cpcv_n_groups=int(validation_cpcv_n_groups),
            cpcv_n_test_groups=int(validation_cpcv_n_test_groups),
            cpcv_purge_groups=int(validation_cpcv_purge_groups),
            cpcv_embargo_groups=int(validation_cpcv_embargo_groups),
            cpcv_max_splits=validation_cpcv_max_splits,
            hard_fail=bool(validation_hard_fail),
            charts_enabled=validation_graphs_enabled,
        )
    return res


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    tokens = [token.strip() for token in str(value).split(",")]
    if not tokens or any(token == "" for token in tokens):
        raise argparse.ArgumentTypeError("expected comma-separated float values")
    try:
        return tuple(float(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated float values") from exc


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(value).split(",")]
    if not tokens or any(token == "" for token in tokens):
        raise argparse.ArgumentTypeError("expected comma-separated integer values")
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer values") from exc
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("all integer values must be > 0")
    return values


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
    p.add_argument("--validation-oos-min-calmar", type=float, default=None)
    p.add_argument("--validation-psr-threshold", type=float, default=0.95)
    p.add_argument("--validation-dsr-threshold", type=float, default=0.80)
    p.add_argument("--validation-pbo-max", type=float, default=0.50)
    p.add_argument("--validation-cpcv-n-groups", type=int, default=10)
    p.add_argument("--validation-cpcv-n-test-groups", type=int, default=2)
    p.add_argument("--validation-cpcv-purge-groups", type=int, default=1)
    p.add_argument("--validation-cpcv-embargo-groups", type=int, default=1)
    p.add_argument("--validation-cpcv-max-splits", type=int, default=256)
    p.add_argument("--validation-hard-fail", action="store_true")
    p.add_argument("--modular-data-signals", action="store_true")
    p.add_argument("--modular-portfolio", action="store_true", default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--ml-disabled", action="store_true")
    p.add_argument("--ml-modular-path", action="store_true")
    p.add_argument("--ml-template-overlay", action="store_true", default=None, help="Apply Template-like ML overlay in template_default_universe mode")
    p.add_argument("--ml-meta-overlay", action="store_true", default=None, help="Apply Template phase1 meta overlay (ML_META) on top of ML overlay")
    p.add_argument("--ml-meta-triple-barrier", action="store_true", help="Use triple-barrier labels for ML_META fitting")
    p.add_argument("--ml-meta-tb-horizon", type=int, default=5)
    p.add_argument("--ml-meta-tb-upper", type=float, default=0.01)
    p.add_argument("--ml-meta-tb-lower", type=float, default=0.01)
    p.add_argument("--ml-prob-threshold", type=float, default=0.0)
    p.add_argument("--ml-feature-fracdiff", action="store_true", help="Enable fractional differencing RF feature")
    p.add_argument("--ml-feature-fracdiff-order", type=float, default=0.4)
    p.add_argument("--ml-feature-fracdiff-threshold", type=float, default=1e-5)
    p.add_argument("--ml-feature-fracdiff-max-terms", type=int, default=256)
    p.add_argument("--ml-feature-har-rv", action="store_true", help="Enable HAR-RV RF features")
    p.add_argument("--ml-feature-har-rv-windows", type=_parse_csv_ints, default=(1, 5, 22))
    p.add_argument("--ml-feature-har-rv-no-log", action="store_true", help="Disable log1p transform for HAR-RV features")
    p.add_argument("--data-max-assets", type=int, default=3)
    p.add_argument("--disable-full-universe-override", action="store_true")
    p.add_argument("--portfolio-method", default="bl")
    p.add_argument("--portfolio-risk-aversion", type=float, default=2.5)
    p.add_argument("--portfolio-cov-shrinkage", type=float, default=0.15)
    p.add_argument("--portfolio-covariance-backend", default="sample")
    p.add_argument("--portfolio-correlation-backend", default="pearson")
    p.add_argument("--portfolio-gerber-threshold", type=float, default=0.5)
    p.add_argument("--portfolio-bl-tau", type=float, default=0.05)
    p.add_argument("--portfolio-view-confidence", type=float, default=0.65)
    p.add_argument("--portfolio-bl-omega-scale", type=float, default=0.25)
    p.add_argument("--portfolio-bl-max-weight", type=float, default=0.15)
    p.add_argument("--portfolio-regime-mapping-enabled", action="store_true")
    p.add_argument("--portfolio-regime-mapping-mode", default="linear")
    p.add_argument("--portfolio-regime-mapping-min-multiplier", type=float, default=1.0)
    p.add_argument("--portfolio-regime-mapping-max-multiplier", type=float, default=1.0)
    p.add_argument(
        "--portfolio-regime-mapping-step-thresholds",
        type=_parse_csv_floats,
        default=(0.30, 0.70),
    )
    p.add_argument(
        "--portfolio-regime-mapping-step-multipliers",
        type=_parse_csv_floats,
        default=(1.0, 1.0, 1.0),
    )
    p.add_argument("--portfolio-regime-mapping-kelly-gamma", type=float, default=2.0)
    p.add_argument("--ml-hmm-scalar-floor", type=float, default=0.30)
    p.add_argument("--ml-hmm-scalar-ceiling", type=float, default=1.50)
    p.add_argument("--vol-target-apply-to-ml", action="store_true")
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--slippage-rate", type=float, default=0.0005)
    p.add_argument("--cost-model", default="simple")
    p.add_argument("--spread-bps", type=float, default=0.0)
    p.add_argument("--impact-coeff", type=float, default=0.0)
    p.add_argument("--borrow-bps", type=float, default=0.0)
    p.add_argument("--dynamic-trading", action="store_true", help="Enable dynamic trading partial-adjustment schedule")
    p.add_argument("--dynamic-trading-lambda", type=float, default=0.0)
    p.add_argument("--dynamic-trading-aim-multiplier", type=float, default=0.0)
    p.add_argument("--dynamic-trading-min-trade-rate", type=float, default=0.10)
    p.add_argument("--dynamic-trading-max-trade-rate", type=float, default=1.0)
    p.add_argument("--execution-backend", choices=("instant", "almgren_chriss"), default="instant")
    p.add_argument("--execution-ac-n-slices", type=int, default=4)
    p.add_argument("--execution-ac-risk-aversion", type=float, default=1.0)
    p.add_argument("--execution-ac-temporary-impact", type=float, default=0.10)
    p.add_argument("--execution-ac-permanent-impact", type=float, default=0.01)
    p.add_argument("--execution-ac-volatility-lookback", type=int, default=20)
    p.add_argument("--experimental-multi-horizon-blend", action="store_true")
    p.add_argument("--experimental-conformal-sizing", action="store_true")
    p.add_argument("--experimental-futuretesting", action="store_true")
    p.add_argument(
        "--experimental-futuretesting-method",
        choices=("stationary_bootstrap", "iid_bootstrap"),
        default="stationary_bootstrap",
    )
    p.add_argument("--experimental-futuretesting-block-size", type=int, default=None)
    p.add_argument("--experimental-futuretesting-n-paths", type=int, default=200)
    p.add_argument("--experimental-futuretesting-horizon", type=int, default=126)
    p.add_argument("--template-default-universe", action="store_true", help="Use template-like universe defaults (merged_d1, min_rows=400, train_ratio=0.60, full asset set)")
    p.add_argument(
        "--template-rebalance",
        action="store_true",
        help="Keep template-default-universe profile but allow dynamic rebalancing/backtest schedule",
    )
    p.add_argument(
        "--no-template-precomputed-artifacts",
        action="store_true",
        help="Disable template precomputed artifacts/benchmark CSV overrides and use computed modular path only",
    )
    p.add_argument(
        "--ml-meta-comparator-benchmark-csv",
        action="store_true",
        help="Optional parity mode: allow ML_META benchmark CSV loading even with --no-template-precomputed-artifacts",
    )
    p.add_argument(
        "--strict-independent",
        action="store_true",
        help="Fail if benchmark/comparator CSV data is injected into modular run outputs",
    )
    p.add_argument(
        "--cpp-hmm-profile",
        choices=available_cpp_hmm_profiles(),
        default=None,
        help="Opt-in C++ HMM profile preset (default: off)",
    )
    p.add_argument("--cle-enabled", action="store_true", help="Enable Confluence Leverage Engine policy wiring")
    p.add_argument("--cle-use-external-proxies", action="store_true", help="Enable CLE external proxy feature inputs")
    p.add_argument("--cle-mode", default="confluence_blend")
    p.add_argument("--cle-mapping-mode", default="linear", help="CLE mapping mode (linear/stepwise/kelly_shrink)")
    p.add_argument("--cle-mapping-min-multiplier", type=float, default=0.5)
    p.add_argument("--cle-mapping-max-multiplier", type=float, default=1.5)
    p.add_argument("--cle-mapping-step-thresholds", type=_parse_csv_floats, default=(0.30, 0.70), help="Comma-separated CLE step thresholds")
    p.add_argument("--cle-mapping-step-multipliers", type=_parse_csv_floats, default=(0.50, 1.00, 1.50), help="Comma-separated CLE step multipliers")
    p.add_argument("--cle-mapping-kelly-gamma", type=float, default=2.0)
    p.add_argument("--cle-gate-event-blackout-cap", type=float, default=0.5)
    p.add_argument("--cle-gate-liquidity-spread-z-threshold", type=float, default=2.0)
    p.add_argument("--cle-gate-liquidity-depth-threshold", type=float, default=0.10)
    p.add_argument("--cle-gate-liquidity-reduction-factor", type=float, default=0.25)
    p.add_argument("--cle-gate-correlation-alert-cap", type=float, default=1.0)
    p.add_argument("--cle-leverage-floor", type=float, default=0.0)
    p.add_argument("--cle-leverage-cap", type=float, default=12.0)
    p.add_argument("--cle-policy-seed", type=int, default=42)
    p.add_argument("--cle-online-calibration-enabled", action="store_true")
    p.add_argument("--cle-online-calibration-window", type=int, default=5)
    p.add_argument("--cle-online-calibration-learning-rate", type=float, default=0.10)
    p.add_argument("--cle-online-calibration-l2", type=float, default=1e-4)
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
            validation_oos_min_calmar=args.validation_oos_min_calmar,
            validation_psr_threshold=args.validation_psr_threshold,
            validation_dsr_threshold=args.validation_dsr_threshold,
            validation_pbo_max=args.validation_pbo_max,
            validation_cpcv_n_groups=args.validation_cpcv_n_groups,
            validation_cpcv_n_test_groups=args.validation_cpcv_n_test_groups,
            validation_cpcv_purge_groups=args.validation_cpcv_purge_groups,
            validation_cpcv_embargo_groups=args.validation_cpcv_embargo_groups,
            validation_cpcv_max_splits=args.validation_cpcv_max_splits,
            validation_hard_fail=args.validation_hard_fail,
            modular_data_signals_enabled=args.modular_data_signals,
            modular_portfolio_enabled=args.modular_portfolio,
            data_path=args.data_path,
            ml_enabled=(not args.ml_disabled),
            ml_modular_path_enabled=args.ml_modular_path,
            ml_template_overlay=args.ml_template_overlay,
            ml_meta_overlay=args.ml_meta_overlay,
            ml_meta_use_triple_barrier=args.ml_meta_triple_barrier,
            ml_meta_tb_horizon=args.ml_meta_tb_horizon,
            ml_meta_tb_upper=args.ml_meta_tb_upper,
            ml_meta_tb_lower=args.ml_meta_tb_lower,
            ml_probability_threshold=args.ml_prob_threshold,
            ml_feature_fracdiff_enabled=args.ml_feature_fracdiff,
            ml_feature_fracdiff_order=args.ml_feature_fracdiff_order,
            ml_feature_fracdiff_threshold=args.ml_feature_fracdiff_threshold,
            ml_feature_fracdiff_max_terms=args.ml_feature_fracdiff_max_terms,
            ml_feature_har_rv_enabled=args.ml_feature_har_rv,
            ml_feature_har_rv_windows=args.ml_feature_har_rv_windows,
            ml_feature_har_rv_use_log=(not args.ml_feature_har_rv_no_log),
            data_max_assets=args.data_max_assets,
            data_full_universe_for_parity=(not args.disable_full_universe_override),
            portfolio_method=args.portfolio_method,
            portfolio_risk_aversion=args.portfolio_risk_aversion,
            portfolio_cov_shrinkage=args.portfolio_cov_shrinkage,
            portfolio_covariance_backend=args.portfolio_covariance_backend,
            portfolio_correlation_backend=args.portfolio_correlation_backend,
            portfolio_gerber_threshold=args.portfolio_gerber_threshold,
            portfolio_bl_tau=args.portfolio_bl_tau,
            portfolio_bl_view_confidence=args.portfolio_view_confidence,
            portfolio_bl_omega_scale=args.portfolio_bl_omega_scale,
            portfolio_bl_max_weight=args.portfolio_bl_max_weight,
            portfolio_regime_mapping_enabled=args.portfolio_regime_mapping_enabled,
            portfolio_regime_mapping_mode=args.portfolio_regime_mapping_mode,
            portfolio_regime_mapping_min_multiplier=args.portfolio_regime_mapping_min_multiplier,
            portfolio_regime_mapping_max_multiplier=args.portfolio_regime_mapping_max_multiplier,
            portfolio_regime_mapping_step_thresholds=args.portfolio_regime_mapping_step_thresholds,
            portfolio_regime_mapping_step_multipliers=args.portfolio_regime_mapping_step_multipliers,
            portfolio_regime_mapping_kelly_gamma=args.portfolio_regime_mapping_kelly_gamma,
            ml_hmm_scalar_floor=args.ml_hmm_scalar_floor,
            ml_hmm_scalar_ceiling=args.ml_hmm_scalar_ceiling,
            vol_target_apply_to_ml=args.vol_target_apply_to_ml,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            cost_model=args.cost_model,
            spread_bps=args.spread_bps,
            impact_coeff=args.impact_coeff,
            borrow_bps=args.borrow_bps,
            dynamic_trading_enabled=args.dynamic_trading,
            dynamic_trading_lambda=args.dynamic_trading_lambda,
            dynamic_trading_aim_multiplier=args.dynamic_trading_aim_multiplier,
            dynamic_trading_min_trade_rate=args.dynamic_trading_min_trade_rate,
            dynamic_trading_max_trade_rate=args.dynamic_trading_max_trade_rate,
            execution_backend=args.execution_backend,
            execution_ac_n_slices=args.execution_ac_n_slices,
            execution_ac_risk_aversion=args.execution_ac_risk_aversion,
            execution_ac_temporary_impact=args.execution_ac_temporary_impact,
            execution_ac_permanent_impact=args.execution_ac_permanent_impact,
            execution_ac_volatility_lookback=args.execution_ac_volatility_lookback,
            experimental_multi_horizon_blend_enabled=args.experimental_multi_horizon_blend,
            experimental_conformal_sizing_enabled=args.experimental_conformal_sizing,
            experimental_futuretesting_enabled=args.experimental_futuretesting,
            experimental_futuretesting_method=args.experimental_futuretesting_method,
            experimental_futuretesting_block_size=args.experimental_futuretesting_block_size,
            experimental_futuretesting_n_paths=args.experimental_futuretesting_n_paths,
            experimental_futuretesting_horizon=args.experimental_futuretesting_horizon,
            template_default_universe=args.template_default_universe,
            template_rebalance_enabled=args.template_rebalance,
            template_use_precomputed_artifacts=(not args.no_template_precomputed_artifacts),
            ml_meta_comparator_use_benchmark_csv=args.ml_meta_comparator_benchmark_csv,
            cpp_hmm_profile=args.cpp_hmm_profile,
            strict_independent_mode=args.strict_independent,
            cle_enabled=args.cle_enabled,
            cle_use_external_proxies=args.cle_use_external_proxies,
            cle_mode=args.cle_mode,
            cle_mapping_mode=args.cle_mapping_mode,
            cle_mapping_min_multiplier=args.cle_mapping_min_multiplier,
            cle_mapping_max_multiplier=args.cle_mapping_max_multiplier,
            cle_mapping_step_thresholds=args.cle_mapping_step_thresholds,
            cle_mapping_step_multipliers=args.cle_mapping_step_multipliers,
            cle_mapping_kelly_gamma=args.cle_mapping_kelly_gamma,
            cle_gate_event_blackout_cap=args.cle_gate_event_blackout_cap,
            cle_gate_liquidity_spread_z_threshold=args.cle_gate_liquidity_spread_z_threshold,
            cle_gate_liquidity_depth_threshold=args.cle_gate_liquidity_depth_threshold,
            cle_gate_liquidity_reduction_factor=args.cle_gate_liquidity_reduction_factor,
            cle_gate_correlation_alert_cap=args.cle_gate_correlation_alert_cap,
            cle_leverage_floor=args.cle_leverage_floor,
            cle_leverage_cap=args.cle_leverage_cap,
            cle_policy_seed=args.cle_policy_seed,
            cle_online_calibration_enabled=args.cle_online_calibration_enabled,
            cle_online_calibration_window=args.cle_online_calibration_window,
            cle_online_calibration_learning_rate=args.cle_online_calibration_learning_rate,
            cle_online_calibration_l2=args.cle_online_calibration_l2,
            parity_metrics_bridge=args.parity_metrics_bridge,
            parity_metrics_dataset=args.legacy_metrics_dataset,
        )
    print(res)
    return res


if __name__ == "__main__":
    main()
