from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .loaders import ExternalProxyLoadResult, load_proxy_from_csv, load_proxy_from_stub


def _empty_result(proxy_name: str, attempted_sources: list[dict[str, Any]]) -> ExternalProxyLoadResult:
    diagnostics: dict[str, Any] = {
        "proxy": proxy_name,
        "status": "missing",
        "used_source": "missing",
        "attempted_sources": attempted_sources,
    }
    return ExternalProxyLoadResult(
        series=pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=proxy_name),
        diagnostics=diagnostics,
    )


def load_proxy_adapter(
    *,
    proxy_name: str,
    csv_path: str | Path | None = None,
    stub_rows: Sequence[Mapping[str, Any]] | None = None,
    date_column_candidates: Sequence[str] = ("date", "datetime", "timestamp"),
    value_column_candidates: Sequence[str] = ("value",),
    timestamp_unit: str = "ms",
) -> ExternalProxyLoadResult:
    attempted: list[dict[str, Any]] = []

    csv_result = load_proxy_from_csv(
        csv_path,
        proxy_name=proxy_name,
        date_column_candidates=date_column_candidates,
        value_column_candidates=value_column_candidates,
        timestamp_unit=timestamp_unit,
    )
    attempted.append(dict(csv_result.diagnostics))
    if csv_result.diagnostics.get("status") == "ok":
        diagnostics = dict(csv_result.diagnostics)
        diagnostics["attempted_sources"] = attempted
        return ExternalProxyLoadResult(series=csv_result.series, diagnostics=diagnostics)

    stub_result = load_proxy_from_stub(
        stub_rows,
        proxy_name=proxy_name,
        date_column_candidates=date_column_candidates,
        value_column_candidates=value_column_candidates,
        timestamp_unit=timestamp_unit,
    )
    attempted.append(dict(stub_result.diagnostics))
    if stub_result.diagnostics.get("status") == "ok":
        diagnostics = dict(stub_result.diagnostics)
        diagnostics["attempted_sources"] = attempted
        return ExternalProxyLoadResult(series=stub_result.series, diagnostics=diagnostics)

    return _empty_result(proxy_name=proxy_name, attempted_sources=attempted)


def load_macro_proxy_adapter(
    csv_path: str | Path | None = None,
    stub_rows: Sequence[Mapping[str, Any]] | None = None,
) -> ExternalProxyLoadResult:
    return load_proxy_adapter(
        proxy_name="macro_proxy",
        csv_path=csv_path,
        stub_rows=stub_rows,
        value_column_candidates=("macro_proxy", "macro", "value"),
    )


def load_calendar_proxy_adapter(
    csv_path: str | Path | None = None,
    stub_rows: Sequence[Mapping[str, Any]] | None = None,
) -> ExternalProxyLoadResult:
    return load_proxy_adapter(
        proxy_name="calendar_proxy",
        csv_path=csv_path,
        stub_rows=stub_rows,
        value_column_candidates=("calendar_proxy", "is_event", "event", "value"),
    )


def load_liquidity_proxy_adapter(
    csv_path: str | Path | None = None,
    stub_rows: Sequence[Mapping[str, Any]] | None = None,
) -> ExternalProxyLoadResult:
    return load_proxy_adapter(
        proxy_name="liquidity_proxy",
        csv_path=csv_path,
        stub_rows=stub_rows,
        value_column_candidates=("liquidity_proxy", "liquidity", "spread", "value"),
    )


def load_cle_external_proxies(
    *,
    macro_csv_path: str | Path | None = None,
    calendar_csv_path: str | Path | None = None,
    liquidity_csv_path: str | Path | None = None,
    macro_stub_rows: Sequence[Mapping[str, Any]] | None = None,
    calendar_stub_rows: Sequence[Mapping[str, Any]] | None = None,
    liquidity_stub_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    macro = load_macro_proxy_adapter(csv_path=macro_csv_path, stub_rows=macro_stub_rows)
    calendar = load_calendar_proxy_adapter(csv_path=calendar_csv_path, stub_rows=calendar_stub_rows)
    liquidity = load_liquidity_proxy_adapter(csv_path=liquidity_csv_path, stub_rows=liquidity_stub_rows)

    panel = pd.concat([macro.series, calendar.series, liquidity.series], axis=1).sort_index()
    panel = panel.reindex(columns=["macro_proxy", "calendar_proxy", "liquidity_proxy"])

    diagnostics = {
        "macro": dict(macro.diagnostics),
        "calendar": dict(calendar.diagnostics),
        "liquidity": dict(liquidity.diagnostics),
    }
    return panel, diagnostics
