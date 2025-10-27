"""Data loading and summarization utilities.

This module keeps heavy imports (pandas) inside functions so the package
can be imported in environments where pandas is not installed (tests may
skip accordingly).
"""
from typing import Any, Dict


def load_excel(path: str, **kwargs) -> Any:
    """Load an Excel file returning a pandas DataFrame.

    The import of pandas is done inside the function to avoid import-time
    failures when pandas is not present (tests can skip).

    Args:
        path: Path to the Excel (.xlsx) file.
        **kwargs: Passed to pandas.read_excel.

    Returns:
        pandas.DataFrame
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required to load Excel files") from exc

    df = pd.read_excel(path, **kwargs)
    return df


def normalize_header(df):
    """Use the first row as header if it appears to be a header row.

    Heuristic: if more than half of the first-row values are non-numeric
    strings (and not empty), treat the first row as the header.

    Returns a new DataFrame with normalized header and reset index.
    """
    import pandas as pd

    if df.shape[0] == 0:
        return df.copy()

    first = df.iloc[0].fillna("")

    # count how many first-row entries are non-numeric strings
    def looks_like_number(x):
        try:
            pd.to_numeric(x)
            return True
        except Exception:
            return False

    str_like = 0
    total = 0
    for v in first:
        total += 1
        s = str(v).strip()
        if s == "":
            continue
        if not looks_like_number(s):
            str_like += 1

    use_header = (str_like / max(total, 1)) >= 0.5

    if not use_header:
        return df.copy()

    # use first row as header
    new_cols = [str(x).strip() if str(x).strip() != "" else f"col_{i}" for i, x in enumerate(first)]

    # ensure unique column names
    seen = {}
    uniq_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            uniq = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
            uniq = c
        uniq_cols.append(uniq)

    new_df = df.copy()
    new_df.columns = uniq_cols
    new_df = new_df.iloc[1:].reset_index(drop=True)
    return new_df


def coerce_numeric(df, exclude_cols=None):
    """Attempt to coerce dataframe columns to numeric where appropriate.

    Args:
        df: pandas.DataFrame
        exclude_cols: iterable of column names to skip

    Returns:
        A copy of df with numeric coercions applied.
    """
    import pandas as pd

    if exclude_cols is None:
        exclude_cols = []

    out = df.copy()
    for col in out.columns:
        if col in exclude_cols:
            continue
        # try coercion
        coerced = pd.to_numeric(out[col], errors="coerce")
        # if coercion produced at least one numeric value and not all NaN,
        # adopt it
        if coerced.notna().sum() > 0:
            out[col] = coerced

    return out


def parse_datetime_columns(df, date_col=None, time_col=None, target_col="timestamp"):
    """Parse date/time columns and optionally combine into single datetime column.

    If date_col is provided, attempts to parse it with pandas.to_datetime.
    If time_col is provided, parse and combine. Returns a copy of df with
    a new column named `target_col` when parsing succeeds.
    """
    import pandas as pd

    out = df.copy()
    if date_col is None:
        return out

    try:
        dates = pd.to_datetime(out[date_col], errors="coerce")
    except Exception:
        dates = pd.Series([pd.NaT] * len(out))

    if time_col is not None and time_col in out.columns:
        try:
            times = pd.to_datetime(out[time_col], format="%H:%M:%S", errors="coerce").dt.time
        except Exception:
            times = pd.Series([None] * len(out))

        # combine
        combined = []
        for d, t in zip(dates, times):
            if pd.isna(d):
                combined.append(pd.NaT)
                continue
            if t is None or pd.isna(t):
                combined.append(d)
            else:
                combined.append(pd.Timestamp(d.date()).replace(hour=t.hour, minute=t.minute, second=t.second))

        out[target_col] = pd.to_datetime(pd.Series(combined))
    else:
        out[target_col] = dates

    return out


def filter_by_value(df, column: str, value) -> Any:
    """Return rows where str(column) equals given value (flexible types).

    The comparison is done by converting both cell and value to string and
    stripping whitespace. This avoids type mismatches between numbers and
    strings (e.g., 5001 vs "5001").

    Args:
        df: pandas.DataFrame
        column: column name to filter on
        value: value to match (int/str)

    Returns:
        Filtered DataFrame (copy)
    """
    import pandas as pd

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    val_s = str(value).strip()

    # Build mask with flexible comparisons: string match OR numeric match
    col_series = df[column]
    # Convert to string first (handles NaN)
    col_str = col_series.astype(str).str.strip()

    # direct string equality
    mask = col_str == val_s

    # try numeric comparison: if value and cell both convertible to float,
    # compare numerically (handles '5001' vs '5001.0')
    try:
        target_num = float(val_s)
        # create a numeric series (NaN if not convertible)
        import pandas as _pd

        col_num = _pd.to_numeric(col_series, errors="coerce")
        num_mask = col_num == target_num
        # combine
        mask = mask | num_mask.fillna(False)
    except Exception:
        # if conversion fails, ignore numeric matching
        pass

    return df[mask].copy()


def summarize_df(df) -> Dict[str, Any]:
    """Return a serializable summary of a pandas DataFrame.

    The summary is a plain dict containing basic stats suitable for
    writing to JSON or printing in a notebook.

    Args:
        df: pandas.DataFrame

    Returns:
        dict with keys: num_rows, num_cols, columns (list of {name,dtype}),
        null_counts, sample_head (list of rows as dicts), describe (numeric).
    """
    # avoid requiring pandas at import time; import locally
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required for summarization") from exc

    if not hasattr(df, "shape"):
        raise TypeError("summarize_df expects a pandas DataFrame-like object")

    num_rows, num_cols = df.shape
    columns = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
    null_counts = df.isnull().sum().to_dict()

    # sample head as records (limit 5)
    sample_head = df.head(5).to_dict(orient="records")

    # descriptive statistics for numeric columns (guard empty case)
    try:
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            describe = {}
        else:
            describe = numeric.describe().to_dict()
    except Exception:
        describe = {}

    # Helper to convert numpy/pandas types and datetimes to plain Python types
    def _make_serializable(o):
        try:
            import numpy as _np
            import pandas as _pd
        except Exception:
            _np = None
            _pd = None

        if isinstance(o, dict):
            return {k: _make_serializable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_make_serializable(v) for v in o]
        # pandas Timestamp
        if _pd is not None and isinstance(o, _pd.Timestamp):
            return o.isoformat()
        # numpy scalars
        if _np is not None and isinstance(o, (_np.integer, _np.int_, _np.intc, _np.intp, _np.int64)):
            return int(o)
        if _np is not None and isinstance(o, (_np.floating, _np.float_, _np.float64)):
            return float(o)
        # datetime and time
        import datetime as _dt

        if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
            return o.isoformat()

        # fallback
        try:
            # some types like numpy arrays are not JSON friendly
            return o.item() if hasattr(o, "item") else o
        except Exception:
            return str(o)

    return {
        "num_rows": int(num_rows),
        "num_cols": int(num_cols),
        "columns": columns,
        "null_counts": {k: int(v) for k, v in null_counts.items()},
        "sample_head": _make_serializable(sample_head),
        "describe": _make_serializable(describe),
    }
