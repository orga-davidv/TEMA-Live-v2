from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ExternalProxyLoadResult:
    series: pd.Series
    diagnostics: dict[str, Any]


def _empty_series(proxy_name: str) -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=proxy_name)


def _pick_column(candidates: Iterable[str], columns: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _normalize_series(date_values: pd.Series, value_values: pd.Series, proxy_name: str, timestamp_unit: str) -> pd.Series:
    date_series = pd.Series(date_values)
    if pd.api.types.is_numeric_dtype(date_series):
        idx = pd.to_datetime(date_series, unit=timestamp_unit, utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(date_series, utc=True, errors="coerce")

    values = pd.to_numeric(value_values, errors="coerce")
    out = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(idx), name=proxy_name)
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna()
    return out


def _diag(
    proxy_name: str,
    source: str,
    status: str,
    used_source: str,
    **kwargs: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "proxy": proxy_name,
        "source": source,
        "status": status,
        "used_source": used_source,
    }
    base.update(kwargs)
    return base


def load_proxy_from_csv(
    csv_path: str | Path | None,
    *,
    proxy_name: str,
    date_column_candidates: Sequence[str] = ("date", "datetime", "timestamp"),
    value_column_candidates: Sequence[str] = ("value",),
    timestamp_unit: str = "ms",
) -> ExternalProxyLoadResult:
    if csv_path is None:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="csv",
                status="missing",
                used_source="missing",
                reason="path_not_provided",
                path=None,
            ),
        )

    path = Path(csv_path)
    if not path.exists():
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="csv",
                status="missing",
                used_source="missing",
                reason="file_not_found",
                path=str(path),
            ),
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="csv",
                status="error",
                used_source="missing",
                reason="read_error",
                path=str(path),
                error=str(exc),
            ),
        )

    if df.empty:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="csv",
                status="empty",
                used_source="missing",
                reason="no_rows",
                path=str(path),
                rows_in=0,
            ),
        )

    value_candidates = tuple(value_column_candidates)
    if proxy_name not in value_candidates:
        value_candidates = (proxy_name, *value_candidates)

    date_col = _pick_column(date_column_candidates, df.columns)
    value_col = _pick_column(value_candidates, df.columns)
    if date_col is None or value_col is None:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="csv",
                status="invalid_schema",
                used_source="missing",
                reason="required_columns_missing",
                path=str(path),
                columns=list(map(str, df.columns)),
                date_column=date_col,
                value_column=value_col,
            ),
        )

    series = _normalize_series(df[date_col], df[value_col], proxy_name=proxy_name, timestamp_unit=timestamp_unit)
    if series.empty:
        return ExternalProxyLoadResult(
            series=series,
            diagnostics=_diag(
                proxy_name,
                source="csv",
                status="empty_after_parse",
                used_source="missing",
                reason="no_valid_rows",
                path=str(path),
                rows_in=int(len(df)),
                rows_loaded=0,
                date_column=date_col,
                value_column=value_col,
            ),
        )

    return ExternalProxyLoadResult(
        series=series,
        diagnostics=_diag(
            proxy_name,
            source="csv",
            status="ok",
            used_source="csv",
            path=str(path),
            rows_in=int(len(df)),
            rows_loaded=int(series.shape[0]),
            date_column=date_col,
            value_column=value_col,
        ),
    )


def load_proxy_from_stub(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    proxy_name: str,
    date_column_candidates: Sequence[str] = ("date", "datetime", "timestamp"),
    value_column_candidates: Sequence[str] = ("value",),
    timestamp_unit: str = "ms",
) -> ExternalProxyLoadResult:
    if rows is None:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="stub",
                status="missing",
                used_source="missing",
                reason="stub_not_provided",
            ),
        )

    if len(rows) == 0:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="stub",
                status="empty",
                used_source="missing",
                reason="stub_empty",
                rows_in=0,
            ),
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="stub",
                status="empty",
                used_source="missing",
                reason="stub_empty",
                rows_in=0,
            ),
        )

    value_candidates = tuple(value_column_candidates)
    if proxy_name not in value_candidates:
        value_candidates = (proxy_name, *value_candidates)

    date_col = _pick_column(date_column_candidates, df.columns)
    value_col = _pick_column(value_candidates, df.columns)
    if date_col is None or value_col is None:
        return ExternalProxyLoadResult(
            series=_empty_series(proxy_name),
            diagnostics=_diag(
                proxy_name,
                source="stub",
                status="invalid_schema",
                used_source="missing",
                reason="required_columns_missing",
                columns=list(map(str, df.columns)),
                date_column=date_col,
                value_column=value_col,
            ),
        )

    series = _normalize_series(df[date_col], df[value_col], proxy_name=proxy_name, timestamp_unit=timestamp_unit)
    if series.empty:
        return ExternalProxyLoadResult(
            series=series,
            diagnostics=_diag(
                proxy_name,
                source="stub",
                status="empty_after_parse",
                used_source="missing",
                reason="no_valid_rows",
                rows_in=int(len(df)),
                rows_loaded=0,
                date_column=date_col,
                value_column=value_col,
            ),
        )

    return ExternalProxyLoadResult(
        series=series,
        diagnostics=_diag(
            proxy_name,
            source="stub",
            status="ok",
            used_source="stub",
            rows_in=int(len(df)),
            rows_loaded=int(series.shape[0]),
            date_column=date_col,
            value_column=value_col,
        ),
    )
