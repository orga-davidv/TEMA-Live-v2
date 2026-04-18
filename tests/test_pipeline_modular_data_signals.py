import json
from pathlib import Path

from tema.config import BacktestConfig
from tema.pipeline import run_pipeline


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_pipeline_uses_modular_data_signals_when_enabled(tmp_path):
    data_dir = tmp_path / "merged_d1"
    out_root = tmp_path / "outputs"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,100\n"
        "1677715200000,101\n"
        "1677801600000,102\n"
        "1678060800000,103\n"
        "1678147200000,104\n",
    )
    _write_csv(
        data_dir / "b_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,50\n"
        "1677715200000,49\n"
        "1677801600000,50\n"
        "1678060800000,51\n"
        "1678147200000,52\n",
    )

    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        data_path=str(data_dir),
        data_max_assets=2,
        data_min_rows=4,
        data_train_ratio=0.8,
        signal_fast_period=2,
        signal_slow_period=3,
        signal_method="ema",
    )
    res = run_pipeline(run_id="modular-ds-test", cfg=cfg, out_root=str(out_root))
    portfolio_info = json.loads((Path(res["out_dir"]) / "portfolio_info.json").read_text(encoding="utf-8"))
    expected_alphas = json.loads((Path(res["out_dir"]) / "expected_alphas.json").read_text(encoding="utf-8"))
    candidate_weights = json.loads((Path(res["out_dir"]) / "candidate_weights.json").read_text(encoding="utf-8"))
    final_weights = json.loads((Path(res["out_dir"]) / "final_weights.json").read_text(encoding="utf-8"))

    assert portfolio_info["enabled"] is True
    assert "rebalance_gate" in portfolio_info
    assert portfolio_info["fallback_used"] is False
    assert portfolio_info["portfolio_modular_enabled"] is True
    assert portfolio_info["portfolio_method"] in {"black_litterman_like", "mean_variance", "mean_variance_fallback"}
    assert len(expected_alphas) == 2
    assert len(candidate_weights) == 2
    assert abs(sum(candidate_weights) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in candidate_weights)
    assert len(final_weights) == 2
    assert abs(sum(final_weights) - 1.0) < 1e-9


def test_pipeline_backtest_uses_test_split_rows(tmp_path):
    data_dir = tmp_path / "merged_d1"
    out_root = tmp_path / "outputs"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,100\n"
        "1677715200000,101\n"
        "1677801600000,102\n"
        "1678060800000,103\n"
        "1678147200000,104\n"
        "1678233600000,105\n",
    )
    _write_csv(
        data_dir / "b_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,50\n"
        "1677715200000,49\n"
        "1677801600000,48\n"
        "1678060800000,49\n"
        "1678147200000,50\n"
        "1678233600000,51\n",
    )
    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        data_path=str(data_dir),
        data_train_ratio=0.5,
        data_min_rows=4,
    )
    res = run_pipeline(run_id="modular-split-test", cfg=cfg, out_root=str(out_root))
    performance = json.loads((Path(res["out_dir"]) / "performance.json").read_text(encoding="utf-8"))

    assert performance["source"]["mode"] == "historical_test_data"
    assert performance["source"]["train_rows"] == 3
    assert performance["source"]["test_rows"] == 3
    assert performance["source"]["rows"] == 2


def test_pipeline_uses_full_universe_override_for_default_cap(tmp_path):
    data_dir = tmp_path / "merged_d1"
    out_root = tmp_path / "outputs"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,100\n"
        "1677715200000,101\n"
        "1677801600000,102\n"
        "1678060800000,103\n",
    )
    _write_csv(
        data_dir / "b_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,90\n"
        "1677715200000,91\n"
        "1677801600000,92\n"
        "1678060800000,93\n",
    )
    _write_csv(
        data_dir / "c_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,80\n"
        "1677715200000,81\n"
        "1677801600000,82\n"
        "1678060800000,83\n",
    )
    _write_csv(
        data_dir / "d_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,70\n"
        "1677715200000,71\n"
        "1677801600000,72\n"
        "1678060800000,73\n",
    )

    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        data_path=str(data_dir),
        data_min_rows=4,
    )
    res = run_pipeline(run_id="modular-full-universe", cfg=cfg, out_root=str(out_root))
    portfolio_info = json.loads((Path(res["out_dir"]) / "portfolio_info.json").read_text(encoding="utf-8"))

    assert portfolio_info["full_universe_override"] is True
    assert len(portfolio_info["assets"]) == 4


def test_pipeline_template_default_universe_applies_template_data_profile(tmp_path):
    data_dir = tmp_path / "merged_d1"
    out_root = tmp_path / "outputs"
    data_dir.mkdir()

    long_rows = ["timestamp,close_mid"]
    for i in range(500):
        ts = 1677628800000 + (i * 86400000)
        long_rows.append(f"{ts},{100 + i}")
    _write_csv(data_dir / "long_d1_merged.csv", "\n".join(long_rows) + "\n")

    short_rows = ["timestamp,close_mid"]
    for i in range(350):
        ts = 1677628800000 + (i * 86400000)
        short_rows.append(f"{ts},{200 + i}")
    _write_csv(data_dir / "short_d1_merged.csv", "\n".join(short_rows) + "\n")

    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        data_path=str(data_dir),
        data_min_rows=4,
        data_train_ratio=0.8,
        template_default_universe=True,
    )
    res = run_pipeline(run_id="template-universe-profile", cfg=cfg, out_root=str(out_root))
    portfolio_info = json.loads((Path(res["out_dir"]) / "portfolio_info.json").read_text(encoding="utf-8"))
    performance = json.loads((Path(res["out_dir"]) / "performance.json").read_text(encoding="utf-8"))

    assert portfolio_info["template_default_universe"] is True
    assert portfolio_info["data_min_rows_used"] == 400
    assert portfolio_info["data_train_ratio_used"] == 0.60
    assert portfolio_info["full_universe_override"] is True
    assert portfolio_info["portfolio_modular_enabled"] is True
    assert portfolio_info["portfolio_modular_effective"] is True
    assert portfolio_info["portfolio_method"] in {"black_litterman_like", "mean_variance", "mean_variance_fallback"}
    assert portfolio_info["expected_alpha_source"] == "strategy_train_returns_geometric_annualized"
    assert portfolio_info["expected_alpha_method"] == "geometric_annualized_per_asset"
    assert portfolio_info["returns_window_source"] == "strategy_train_returns"
    assert portfolio_info["split_mode"] == "per_asset"
    assert portfolio_info["assets"] == ["long"]
    assert performance["source"]["train_rows"] == 300
    assert performance["source"]["test_rows"] == 200
    assert performance["source"]["rows"] == 200
    assert performance["source"]["returns_source"] == "strategy_test_returns"
    assert performance["source"]["strategy_returns_include_costs"] is True
    assert performance["source"]["split_mode"] == "per_asset"
