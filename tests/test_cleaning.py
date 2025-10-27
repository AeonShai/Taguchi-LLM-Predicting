import pytest


def test_normalize_header_and_coerce():
    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not installed; skipping cleaning tests")

    from src.data_processing import normalize_header, coerce_numeric

    raw = pd.DataFrame([
        ["Id", "Val", "Date"],
        [1, "10", "2025-01-01"],
        [2, "20", "2025-01-02"],
    ])

    df_norm = normalize_header(raw)
    assert list(df_norm.columns) == ["Id", "Val", "Date"]
    df_coerced = coerce_numeric(df_norm, exclude_cols=["Date"]) 
    assert pd.api.types.is_integer_dtype(df_coerced["Id"].dtype) or pd.api.types.is_float_dtype(df_coerced["Id"].dtype)
    assert pd.api.types.is_float_dtype(df_coerced["Val"].dtype) or pd.api.types.is_integer_dtype(df_coerced["Val"].dtype)


def test_filter_by_value():
    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not installed; skipping filter test")

    from src.data_processing import filter_by_value

    df = pd.DataFrame({"MouldCode": ["5001", "5002", 5001, None], "v": [1, 2, 3, 4]})
    subset = filter_by_value(df, "MouldCode", "5001")
    # should match the string and numeric 5001 -> two rows
    assert len(subset) == 2
