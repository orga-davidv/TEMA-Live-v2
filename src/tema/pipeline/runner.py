from typing import List, Sequence, Optional
from ..config import BacktestConfig
from ..turnover import apply_rebalance_gating
import json
import os
from datetime import datetime


def _portfolio_stage() -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
    """Simplified BL/portfolio stage producing current, candidate, and expected alphas.
    In real code this would call into portfolio/optimization modules. Here we keep
    deterministic, small arrays so orchestration can be tested.
    """
    current = [0.30, 0.40, 0.30]
    candidate = [0.25, 0.45, 0.30]
    expected_alphas = [0.01, 0.02, 0.005]
    return current, candidate, expected_alphas


def _ml_filter_and_scalar(cfg: BacktestConfig, expected_alphas: Sequence[float]) -> dict:
    """Minimal ML stage: optionally adjusts expected_alphas or returns a scalar.
    We return a small dict describing ML decisions to include in the manifest.
    """
    ml_info = {
        "ml_enabled": bool(cfg.ml_enabled),
        "scalar": [1.0 for _ in expected_alphas],
        "notes": "simple-pass-through scalar for Wave 2 smoke runner",
    }
    return ml_info


def _scaling_stage(weights: Sequence[float], ml_info: dict, cfg: BacktestConfig) -> List[float]:
    """Apply ml scalar and a naive vol-target style normalization.
    This keeps deterministic behavior while demonstrating the interface.
    """
    scalar = ml_info.get("scalar", [1.0] * len(weights))
    # validate lengths to avoid silently dropping assets
    if len(scalar) != len(weights):
        raise ValueError(f"Scalar length {len(scalar)} does not match weights length {len(weights)}")
    scaled = [w * s for w, s in zip(weights, scalar)]
    # ensure no negative and normalize to sum 1 unless all zeros
    total = sum(abs(x) for x in scaled)
    if total == 0:
        return list(scaled)
    normalized = [x / total for x in scaled]
    return normalized


def run_pipeline(run_id: Optional[str] = None, cfg: Optional[BacktestConfig] = None, out_root: str = "outputs") -> dict:
    """Execute Wave 2 simplified pipeline and write artifacts under outputs/{run_id}/.

    Returns a dict summary which is also written to manifest.json.
    """
    if run_id is None:
        run_id = datetime.utcnow().strftime("run-%Y%m%dT%H%M%SZ")
    if cfg is None:
        cfg = BacktestConfig()

    # sanitize run_id to avoid path traversal
    import re as _re
    # basic token check
    if not _re.match(r'^[A-Za-z0-9._-]+$', run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")
    # reject single or double-dot ids which can escape directories
    if run_id in ('.', '..'):
        raise ValueError("Invalid run_id; '.' and '..' are not allowed")
    # ensure resolved path remains under out_root to prevent path traversal
    out_root_abs = os.path.abspath(out_root)
    candidate = os.path.abspath(os.path.join(out_root_abs, run_id))
    if not (candidate == out_root_abs or candidate.startswith(out_root_abs + os.sep)):
        raise ValueError("Invalid run_id; resolved path escapes out_root")

    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1: Portfolio (BL-like)
    current, candidate, expected_alphas = _portfolio_stage()

    # Stage 2: ML filter / scaler
    ml_info = _ml_filter_and_scalar(cfg, expected_alphas)

    # Stage 3: Turnover / cost gate
    gated = apply_rebalance_gating(current, candidate, expected_alphas, cfg)

    # Stage 4: Scaling stage
    final_weights = _scaling_stage(gated, ml_info, cfg)

    # Stage 5: Reporting artifacts
    artifacts = {
        "current_weights": current,
        "candidate_weights": candidate,
        "expected_alphas": expected_alphas,
        "gated_weights": gated,
        "final_weights": final_weights,
        "ml_info": ml_info,
    }

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "artifacts": list(artifacts.keys()),
    }

    # write artifacts
    for name, value in artifacts.items():
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)

    # write manifest
    mf_path = os.path.join(out_dir, "manifest.json")
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest_path": mf_path, "out_dir": out_dir, "manifest": manifest}


if __name__ == "__main__":
    # quick CLI for ad-hoc local runs
    import argparse
    parser = argparse.ArgumentParser("tema-pipeline-runner")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    print(run_pipeline(run_id=args.run_id))
