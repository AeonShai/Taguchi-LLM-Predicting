"""Create a subset CSV and EDA summary for a specific MouldCode value.

Usage: run from repository root. It will prefer the cleaned CSV in
`outputs/ham_veri_cleaned.csv` if present; otherwise it will load and clean
the raw Excel file.
"""
from pathlib import Path
import json
import sys

from src.data_processing import load_excel, normalize_header, coerce_numeric, parse_datetime_columns, filter_by_value, summarize_df


DATA_CLEANED = Path("outputs") / "ham_veri_cleaned.csv"
RAW_EXCEL = Path("Veri Analizi") / "Ham_Veri.xlsx"
OUT_DIR = Path("outputs")


def load_or_clean():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_CLEANED.exists():
        import pandas as pd

        df = pd.read_csv(DATA_CLEANED)
        return df

    if not RAW_EXCEL.exists():
        print("No data file found.")
        sys.exit(1)

    df = load_excel(str(RAW_EXCEL))
    df = normalize_header(df)
    # simple exclusion heuristic
    exclude = list(df.columns[-5:]) if df.shape[1] >= 5 else []
    df = coerce_numeric(df, exclude_cols=exclude)
    # parse timestamp if possible
    date_col = None
    time_col = None
    for candidate in ["DateOfLine", "Date", "date"]:
        if candidate in df.columns:
            date_col = candidate
            break
    for candidate in ["ActualTimeOfLine", "Time", "time"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if date_col:
        df = parse_datetime_columns(df, date_col=date_col, time_col=time_col, target_col="timestamp")

    df.to_csv(DATA_CLEANED, index=False, encoding="utf-8")
    return df


def main(mould_value="5001"):
    df = load_or_clean()
    # filter
    try:
        subset = filter_by_value(df, "MouldCode", mould_value)
    except KeyError:
        print("Column 'MouldCode' not found in data")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"ham_veri_mould_{mould_value}.csv"
    subset.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Wrote subset CSV to: {out_csv.resolve()} (rows={len(subset)})")

    summary = summarize_df(subset)
    out_json = OUT_DIR / f"eda_summary_mould_{mould_value}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote subset EDA summary to: {out_json.resolve()}")


if __name__ == "__main__":
    val = "5001"
    if len(sys.argv) > 1:
        val = sys.argv[1]
    main(val)
