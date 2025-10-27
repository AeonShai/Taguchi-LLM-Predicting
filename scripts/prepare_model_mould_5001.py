"""Prepare model-ready CSV for MouldCode=5001 by dropping date/time metadata and selected categorical columns.

Drops:
- any `ts_` derived timestamp features added earlier
- `DateOfLine` and other date-like raw columns
- `MouldName`, `MouldCode`, `OPERATOR` (and `CODE` as it's metadata)

Keeps: `MeasuredCycleDuration`, `DosingTime`, `CoolingTime` and all numeric sensor columns.

Writes: `outputs/ham_veri_mould_5001_model_ready.csv` and a small JSON summary.

Run:
    $env:PYTHONPATH='.'; python scripts/prepare_model_mould_5001.py
"""
from pathlib import Path
import json
import pandas as pd

IN = Path('outputs/ham_veri_mould_5001_cleaned.csv')
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)


def main():
    if not IN.exists():
        print(f"Missing input: {IN}. Run cleaning first.")
        return

    df = pd.read_csv(IN)
    orig_shape = df.shape

    # columns to drop
    drop_prefixes = ['ts_']
    drop_exact = ['DateOfLine', 'MouldName', 'MouldCode', 'OPERATOR', 'CODE']

    cols_to_drop = [c for c in df.columns if any(c.startswith(p) for p in drop_prefixes)]
    cols_to_drop += [c for c in drop_exact if c in df.columns]

    # Also drop any column that is plain 'timestamp' if present
    if 'timestamp' in df.columns:
        cols_to_drop.append('timestamp')

    cols_to_drop = sorted(set(cols_to_drop))

    # Ensure we don't drop critical columns
    keep_cols = {'MeasuredCycleDuration', 'DosingTime', 'CoolingTime'}
    for kc in keep_cols:
        if kc not in df.columns:
            print(f"Warning: expected keep column {kc} not found in data")

    df_model = df.drop(columns=cols_to_drop, errors='ignore')

    out_csv = OUT / 'ham_veri_mould_5001_model_ready.csv'
    df_model.to_csv(out_csv, index=False)

    summary = {
        'orig_shape': orig_shape,
        'model_shape': df_model.shape,
        'dropped_columns': cols_to_drop,
        'model_csv': str(out_csv)
    }
    out_json = OUT / 'prepare_model_mould_5001_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote model-ready CSV: {out_csv} (rows={df_model.shape[0]}, cols={df_model.shape[1]})")
    print(f"Dropped columns: {cols_to_drop}")


if __name__ == '__main__':
    main()
