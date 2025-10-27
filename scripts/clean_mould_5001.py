"""Clean the MouldCode=5001 subset for feature engineering.

Steps performed:
- Load `outputs/ham_veri_mould_5001.csv`
- Drop obvious ID columns: Id, NumberOfLine, ActualTimeOfLine (if present)
- Parse timestamp (try `timestamp`, else combine `DateOfLine` + `ActualTimeOfLine`), extract hour, minute, dayofweek, elapsed_seconds
- Drop original timestamp/date/time columns after extraction
- Impute numeric columns with median
- Save cleaned CSV to `outputs/ham_veri_mould_5001_cleaned.csv` and a JSON summary

Run with:
    $env:PYTHONPATH='.'; python scripts/clean_mould_5001.py
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

IN = Path("outputs/ham_veri_mould_5001.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def try_parse_timestamp(df: pd.DataFrame):
    # Prefer 'timestamp' column
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if ts.notna().sum() > 0:
                return ts, ["timestamp"]
        except Exception:
            pass

    # Try to combine DateOfLine + ActualTimeOfLine
    if "DateOfLine" in df.columns and "ActualTimeOfLine" in df.columns:
        try:
            combined = df["DateOfLine"].astype(str).str.strip() + " " + df["ActualTimeOfLine"].astype(str).str.strip()
            ts = pd.to_datetime(combined, errors="coerce")
            if ts.notna().sum() > 0:
                return ts, ["DateOfLine", "ActualTimeOfLine"]
        except Exception:
            pass

    # fallback: try any column name containing 'date' or 'time' or 'timestamp'
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time", "timestamp")):
            try:
                ts = pd.to_datetime(df[c], errors="coerce")
                if ts.notna().sum() > 0:
                    return ts, [c]
            except Exception:
                continue

    return None, []


def main():
    if not IN.exists():
        print(f"Input missing: {IN}. Create the subset first.")
        return

    df = pd.read_csv(IN)
    summary = {"orig_rows": len(df), "orig_cols": len(df.columns)}

    # Drop ID-like columns
    drop_candidates = ["Id", "NumberOfLine", "ActualTimeOfLine"]
    dropped = [c for c in drop_candidates if c in df.columns]
    df = df.drop(columns=dropped, errors="ignore")
    summary["dropped_columns_initial"] = dropped

    # Parse timestamp
    ts_series, ts_source_cols = try_parse_timestamp(df)
    if ts_series is not None:
        df["_ts"] = ts_series
        parsed = ts_series.notna().sum()
        summary["timestamp_parsed_rows"] = int(parsed)
        # extract features
        df["ts_hour"] = df["_ts"].dt.hour
        df["ts_minute"] = df["_ts"].dt.minute
        df["ts_dayofweek"] = df["_ts"].dt.dayofweek
        # elapsed seconds from min timestamp (numeric)
        ts_min = df["_ts"].min()
        df["ts_elapsed_seconds"] = (df["_ts"] - ts_min).dt.total_seconds()
        # drop original timestamp columns
        df = df.drop(columns=ts_source_cols, errors="ignore")
        # optionally drop internal _ts to keep only features
        df = df.drop(columns=["_ts"], errors="ignore")
        summary["timestamp_source_columns"] = ts_source_cols
    else:
        summary["timestamp_parsed_rows"] = 0
        summary["timestamp_source_columns"] = []

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary["numeric_columns_count"] = len(numeric_cols)

    # Record missing before imput
    missing_before = df[numeric_cols].isna().sum().to_dict() if numeric_cols else {}
    summary["missing_before_numeric_sample"] = {k: int(v) for k, v in list(missing_before.items())[:10]}

    # Impute numeric columns with median
    for c in numeric_cols:
        median = df[c].median()
        if pd.isna(median):
            # column all-NaN — leave as is
            continue
        df[c] = df[c].fillna(median)

    missing_after = df[numeric_cols].isna().sum().to_dict() if numeric_cols else {}
    summary["missing_after_numeric_sample"] = {k: int(v) for k, v in list(missing_after.items())[:10]}

    # Save cleaned CSV
    out_csv = OUT / "ham_veri_mould_5001_cleaned.csv"
    df.to_csv(out_csv, index=False)
    summary["cleaned_rows"] = len(df)
    summary["cleaned_cols"] = len(df.columns)
    summary["cleaned_csv"] = str(out_csv)

    # Save summary
    out_json = OUT / "clean_mould_5001_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote cleaned CSV: {out_csv} (rows={summary['cleaned_rows']}, cols={summary['cleaned_cols']})")
    print(f"Wrote summary: {out_json}")


if __name__ == "__main__":
    main()
