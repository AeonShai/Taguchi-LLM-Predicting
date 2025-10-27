"""Show value counts for MouldCode and CODE to help pick subsets."""
from pathlib import Path
import json
import sys

from src.data_processing import load_excel, normalize_header, coerce_numeric, parse_datetime_columns


DATA_CLEANED = Path("outputs") / "ham_veri_cleaned.csv"
RAW_EXCEL = Path("Veri Analizi") / "Ham_Veri.xlsx"


def load_or_clean():
    if DATA_CLEANED.exists():
        import pandas as pd
        return pd.read_csv(DATA_CLEANED)

    if not RAW_EXCEL.exists():
        print("No data file found.")
        sys.exit(1)

    df = load_excel(str(RAW_EXCEL))
    df = normalize_header(df)
    exclude = list(df.columns[-5:]) if df.shape[1] >= 5 else []
    df = coerce_numeric(df, exclude_cols=exclude)
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

    return df


def main():
    df = load_or_clean()
    for col in ["MouldCode", "CODE", "MPS"]:
        if col in df.columns:
            print(f"\nValue counts for {col} (top 30):")
            vc = df[col].astype(str).str.strip().value_counts(dropna=True)
            print(vc.head(30).to_string())
        else:
            print(f"\nColumn {col} not found in data")


if __name__ == "__main__":
    main()
