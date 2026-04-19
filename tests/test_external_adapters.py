from pathlib import Path

import pytest

from tema.external import (
    load_calendar_proxy_adapter,
    load_cle_external_proxies,
    load_liquidity_proxy_adapter,
    load_macro_proxy_adapter,
    load_proxy_from_csv,
)


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_proxy_from_csv_success_and_diagnostics(tmp_path):
    fp = tmp_path / "macro_proxy.csv"
    _write_csv(
        fp,
        "date,macro_proxy\n"
        "2024-01-01,1.1\n"
        "2024-01-02,1.3\n"
        "2024-01-03,1.2\n",
    )

    result = load_proxy_from_csv(fp, proxy_name="macro_proxy")
    assert list(result.series.astype(float)) == [1.1, 1.3, 1.2]
    assert result.diagnostics["status"] == "ok"
    assert result.diagnostics["used_source"] == "csv"
    assert result.diagnostics["rows_loaded"] == 3


def test_macro_adapter_missing_file_returns_safe_empty_and_diagnostics(tmp_path):
    result = load_macro_proxy_adapter(csv_path=tmp_path / "missing_macro.csv")
    assert result.series.empty
    assert result.diagnostics["used_source"] == "missing"
    assert result.diagnostics["status"] == "missing"
    assert len(result.diagnostics["attempted_sources"]) == 2
    assert result.diagnostics["attempted_sources"][0]["source"] == "csv"
    assert result.diagnostics["attempted_sources"][0]["status"] == "missing"


def test_macro_adapter_falls_back_to_stub_when_csv_missing(tmp_path):
    stub_rows = [
        {"date": "2024-02-01", "value": 2.0},
        {"date": "2024-02-02", "value": 2.5},
    ]
    result = load_macro_proxy_adapter(csv_path=tmp_path / "missing_macro.csv", stub_rows=stub_rows)

    assert list(result.series.astype(float)) == [2.0, 2.5]
    assert result.diagnostics["used_source"] == "stub"
    assert result.diagnostics["status"] == "ok"
    assert len(result.diagnostics["attempted_sources"]) == 2
    assert result.diagnostics["attempted_sources"][0]["source"] == "csv"
    assert result.diagnostics["attempted_sources"][1]["source"] == "stub"


@pytest.mark.parametrize(
    ("adapter", "proxy_name"),
    [
        (load_calendar_proxy_adapter, "calendar_proxy"),
        (load_liquidity_proxy_adapter, "liquidity_proxy"),
    ],
)
def test_specific_adapters_accept_stub_api_shape(adapter, proxy_name):
    stub_rows = [
        {"datetime": "2024-03-01T00:00:00Z", proxy_name: 1.0},
        {"datetime": "2024-03-02T00:00:00Z", proxy_name: 0.0},
    ]
    result = adapter(stub_rows=stub_rows)
    assert list(result.series.astype(float)) == [1.0, 0.0]
    assert result.diagnostics["status"] == "ok"
    assert result.diagnostics["used_source"] == "stub"


def test_load_cle_external_proxies_returns_panel_and_per_proxy_diagnostics(tmp_path):
    macro_csv = tmp_path / "macro.csv"
    _write_csv(
        macro_csv,
        "date,macro_proxy\n"
        "2024-04-01,1.0\n"
        "2024-04-02,1.1\n",
    )

    panel, diagnostics = load_cle_external_proxies(
        macro_csv_path=macro_csv,
        calendar_csv_path=tmp_path / "missing_calendar.csv",
        liquidity_stub_rows=[{"date": "2024-04-01", "value": 0.5}],
    )

    assert list(panel.columns) == ["macro_proxy", "calendar_proxy", "liquidity_proxy"]
    assert diagnostics["macro"]["used_source"] == "csv"
    assert diagnostics["calendar"]["used_source"] == "missing"
    assert diagnostics["liquidity"]["used_source"] == "stub"
