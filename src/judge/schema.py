"""
Dataset summary extraction for the LLM judge (Step 2, §7 of the design doc).

The judge does not need the full dataset — only a compact, JSON-serializable
summary of its shape, columns, and a few sample rows.  This lets the judge
detect nonexistent columns, wrong dtypes, missing-value issues, and claims
unsupported by the available data.

This module is intentionally pure: no caching, no global state.  All numeric
values are cast to native Python ``int``/``float`` so the output never leaks
numpy types into ``json.dumps``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pdt


def _to_py(value: Any) -> Any:
    """Cast a (possibly numpy/pandas) scalar to a JSON-serializable Python value.

    Returns ``None`` for missing values (NaN/NaT/None).  numpy integer/float
    types become native ``int``/``float``; numpy bool becomes ``bool``;
    everything else is coerced to ``str`` as a best-effort fallback.
    """
    if value is None:
        return None
    # pd.isna can raise/return arrays for some inputs; guard with a try.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # Native python types pass through unchanged.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, str):
        return value

    # numpy scalars expose .item() to produce a native python value.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_py(item())
        except (ValueError, TypeError):
            pass

    return str(value)


def _round4(value: Any) -> Any:
    """Round a numeric value to 4 decimals, returning ``None`` for non-finite."""
    py = _to_py(value)
    if py is None:
        return None
    try:
        f = float(py)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def _summarize_column(
    series: pd.Series,
    max_sample_values: int,
) -> dict:
    """Build the per-column summary dict for a single dataframe column."""
    col: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "num_missing": int(series.isna().sum()),
        "num_unique": int(series.nunique(dropna=True)),
    }

    if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
        col["min"] = _round4(series.min())
        col["max"] = _round4(series.max())
        col["mean"] = _round4(series.mean())
    else:
        # object / categorical / bool / datetime: show a few sample values.
        sample = (
            series.dropna()
            .unique()[:max_sample_values]
        )
        col["sample_values"] = [_to_py(v) for v in sample]

    return col


def summarize_csv(
    csv_path: Path | str,
    max_sample_rows: int = 3,
    max_sample_values: int = 3,
) -> dict:
    """Return a compact JSON-serializable summary of a CSV for the judge.

    Keys: ``num_rows``, ``num_columns``, ``columns`` (list of per-column dicts),
    ``sample_rows`` (list of dicts).

    Per-column dict: ``name``, ``dtype``, ``num_missing``, ``num_unique``, and
    either ``sample_values`` (for object/categorical) or ``min``/``max``/``mean``
    (for numeric).

    All values are JSON-serializable (numpy types are cast to python
    ``float``/``int``).  Numeric ``min``/``max``/``mean`` are rounded to 4
    decimal places.

    On an empty, missing, or unreadable file, returns ``{"error": "..."}``
    instead of raising.
    """
    if csv_path is None:
        return {"error": "csv_path is None"}

    path = Path(csv_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"error": f"Empty CSV file (no columns/data): {path}"}
    except Exception as exc:  # noqa: BLE001 - best-effort, surface as error dict
        return {"error": f"Failed to read CSV {path}: {type(exc).__name__}: {exc}"}

    columns = [
        _summarize_column(df[name], max_sample_values=max_sample_values)
        for name in df.columns
    ]

    # Sample rows: convert to records and sanitize every cell.
    sample_records = df.head(max_sample_rows).to_dict(orient="records")
    sample_rows = [
        {str(k): _to_py(v) for k, v in record.items()}
        for record in sample_records
    ]

    return {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": columns,
        "sample_rows": sample_rows,
    }
