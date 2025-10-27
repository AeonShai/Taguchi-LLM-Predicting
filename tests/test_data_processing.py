import pytest


def pytest_configured():
    # nothing to configure for now
    return


def test_summarize_df_smoke():
    """Smoke test for summarize_df: skip if pandas not available."""
    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not installed; skipping DataFrame tests")

    from src.data_processing import summarize_df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, None, 0.3]})
    summary = summarize_df(df)

    assert summary["num_rows"] == 3
    assert summary["num_cols"] == 2
    assert "columns" in summary and isinstance(summary["columns"], list)
    assert "null_counts" in summary
