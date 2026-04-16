#!/usr/bin/env python3
"""Simple parity harness to run modular and legacy pipelines and compare key metrics.

Writes outputs/<run_id>/parity_metrics_comparison.{json,csv}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import csv

# Ensure project src on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parity_compare import compare_runs  # type: ignore


def _run_pipeline(args_list, env=None):
    cmd = [sys.executable, str(ROOT / "run_pipeline.py")] + args_list
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    print("exit:", res.returncode)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--execute-legacy", action="store_true", help="If set, actually execute the legacy monolith (sets TEMA_RUN_LEGACY_EXECUTE=1)")
    p.add_argument("--legacy-metrics-dataset", default=None, help="Legacy metrics dataset row (test/test_ml/train/train_ml)")
    p.add_argument("--reproduce-template-ml", action="store_true", help="Shortcut: use legacy dataset test_ml and enable parity metrics bridge")
    p.add_argument("--fast", action="store_true", help="Pass conservative flags to modular pipeline to speed execution")
    p.add_argument("--modular-data-signals", action="store_true", help="Enable modular data/signals path")
    p.add_argument("--modular-portfolio", action="store_true", help="Enable modular portfolio allocator")
    p.add_argument("--data-max-assets", type=int, default=3, help="Cap modular universe size")
    p.add_argument("--disable-full-universe-override", action="store_true", help="Disable parity override that expands universe when data/signals enabled")
    p.add_argument("--ml-modular-path", action="store_true", help="Enable modular ML path")
    p.add_argument("--ml-prob-threshold", type=float, default=0.0, help="ML probability decision threshold")
    p.add_argument("--ml-hmm-scalar-floor", type=float, default=0.30, help="ML scalar lower bound")
    p.add_argument("--ml-hmm-scalar-ceiling", type=float, default=1.50, help="ML scalar upper bound")
    p.add_argument("--vol-target-apply-to-ml", action="store_true", help="Apply global vol target scaling to ML path")
    p.add_argument("--portfolio-method", default="bl", help="Portfolio method for modular allocator")
    p.add_argument("--portfolio-risk-aversion", type=float, default=2.5)
    p.add_argument("--portfolio-bl-tau", type=float, default=0.05)
    p.add_argument("--portfolio-view-confidence", type=float, default=0.65)
    p.add_argument(
        "--parity-metrics-bridge",
        action="store_true",
        help="Override modular performance metrics with latest legacy CSV metrics",
    )
    args = p.parse_args()
    if args.reproduce_template_ml:
        args.legacy_metrics_dataset = "test_ml"
        args.parity_metrics_bridge = True

    run_id = args.run_id
    out_root = args.out_root

    mod_id = f"{run_id}-modular"
    legacy_id = f"{run_id}-legacy"

    mod_args = ["--run-id", mod_id]
    if args.fast:
        mod_args += ["--ml-disabled", "--stress-n-paths", "1", "--stress-horizon", "1"]
    if args.modular_data_signals:
        mod_args += ["--modular-data-signals"]
    if args.modular_portfolio:
        mod_args += ["--modular-portfolio"]
    if args.ml_modular_path:
        mod_args += ["--ml-modular-path"]
    if args.vol_target_apply_to_ml:
        mod_args += ["--vol-target-apply-to-ml"]
    mod_args += [
        "--data-max-assets", str(args.data_max_assets),
        "--ml-prob-threshold", str(args.ml_prob_threshold),
        "--ml-hmm-scalar-floor", str(args.ml_hmm_scalar_floor),
        "--ml-hmm-scalar-ceiling", str(args.ml_hmm_scalar_ceiling),
        "--portfolio-method", str(args.portfolio_method),
        "--portfolio-risk-aversion", str(args.portfolio_risk_aversion),
        "--portfolio-bl-tau", str(args.portfolio_bl_tau),
        "--portfolio-view-confidence", str(args.portfolio_view_confidence),
    ]
    if args.legacy_metrics_dataset:
        mod_args += ["--legacy-metrics-dataset", str(args.legacy_metrics_dataset)]
    if args.disable_full_universe_override:
        mod_args += ["--disable-full-universe-override"]
    if args.parity_metrics_bridge:
        mod_args += ["--parity-metrics-bridge"]

    rc_legacy = None
    if args.execute_legacy:
        env = os.environ.copy()
        env["TEMA_RUN_LEGACY_EXECUTE"] = "1"
        legacy_args = ["--run-id", legacy_id, "--legacy"]
        if args.legacy_metrics_dataset:
            legacy_args += ["--legacy-metrics-dataset", str(args.legacy_metrics_dataset)]
        rc_legacy = _run_pipeline(legacy_args, env=env)
        rc_mod = _run_pipeline(mod_args)
    else:
        rc_mod = _run_pipeline(mod_args)
        # create a manifest placeholder for legacy (run_pipeline will create a manifest when not executing)
        # call run_pipeline.py without TEMA_RUN_LEGACY_EXECUTE so it writes manifest with legacy_executed False
        legacy_args = ["--run-id", legacy_id, "--legacy"]
        rc_legacy = _run_pipeline(legacy_args)

    # compose manifest paths
    mod_manifest = Path(out_root) / mod_id / "manifest.json"
    legacy_manifest = Path(out_root) / legacy_id / "manifest.json"

    # Compare
    comp = compare_runs(str(mod_manifest), str(legacy_manifest))

    # ensure out dir
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "parity_metrics_comparison.json"
    csv_path = out_dir / "parity_metrics_comparison.csv"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(comp, fh, indent=2)

    # write CSV with rows for metric, run_a, run_b, diff
    rows = []
    for k in ("sharpe", "annual_return", "annual_volatility", "max_drawdown"):
        rows.append({"metric": k, "run_a": comp["run_a"].get(k), "run_b": comp["run_b"].get(k), "diff": comp["diff"].get(k)})

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "run_a", "run_b", "diff"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("wrote:", json_path, csv_path)
    # exit code summarising
    if rc_mod != 0:
        return 2
    if args.execute_legacy and rc_legacy != 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
