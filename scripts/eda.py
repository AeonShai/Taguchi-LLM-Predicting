"""Small EDA runner script.

Usage: run from repository root. It will load `Veri Analizi/Ham_Veri.xlsx`,
summarize it and write `outputs/eda_summary.json`.
"""
import json
import os
from pathlib import Path

from src.data_processing import load_excel, summarize_df, normalize_header, coerce_numeric, parse_datetime_columns


DATA_PATH = Path("Veri Analizi") / "Ham_Veri.xlsx"
OUT_DIR = Path("outputs")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH.resolve()}")
        return

    print(f"Loading data from: {DATA_PATH}")
    df = load_excel(str(DATA_PATH))

    # initial raw summary
    summary = summarize_df(df)
    out_file = OUT_DIR / "eda_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote EDA summary to: {out_file.resolve()}")

    # Cleaning: normalize header and coerce numeric types
    print("Normalizing header (if needed) and coercing numeric columns...")
    df_clean = normalize_header(df)

    # try to identify likely metadata columns to exclude from numeric coercion
    # simple heuristic: treat last 5 columns with many strings as categorical
    exclude = []
    if df_clean.shape[1] >= 5:
        exclude = list(df_clean.columns[-5:])

    df_clean = coerce_numeric(df_clean, exclude_cols=exclude)

    # try parse date/time fields if present
    # common names inferred from header row
    date_col = None
    time_col = None
    for candidate in ["DateOfLine", "Date", "date"]:
        if candidate in df_clean.columns:
            date_col = candidate
            break
    for candidate in ["ActualTimeOfLine", "Time", "time"]:
        if candidate in df_clean.columns:
            time_col = candidate
            break

    if date_col is not None:
        df_clean = parse_datetime_columns(df_clean, date_col=date_col, time_col=time_col, target_col="timestamp")

    # save cleaned CSV
    cleaned_file = OUT_DIR / "ham_veri_cleaned.csv"
    df_clean.to_csv(cleaned_file, index=False, encoding="utf-8")
    print(f"Wrote cleaned CSV to: {cleaned_file.resolve()}")

    # summary after cleaning
    cleaned_summary = summarize_df(df_clean)
    cleaned_out = OUT_DIR / "eda_summary_cleaned.json"
    with open(cleaned_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote cleaned EDA summary to: {cleaned_out.resolve()}")


if __name__ == "__main__":
    main()
