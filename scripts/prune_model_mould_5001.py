"""Prune model-ready CSV by removing zero-variance columns and MPS.

Writes:
- `outputs/ham_veri_mould_5001_pruned.csv`
- `outputs/prune_model_mould_5001_summary.json`

Run:
    $env:PYTHONPATH='.'; python scripts/prune_model_mould_5001.py
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

IN = Path('outputs/ham_veri_mould_5001_model_ready.csv')
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)


def main():
    if not IN.exists():
        print(f"Missing input: {IN}. Run prepare_model_mould_5001.py first.")
        return

    df = pd.read_csv(IN)
    orig_shape = df.shape

    # find zero-variance (unique == 1) columns
    unique_counts = df.nunique(dropna=False)
    zero_var_cols = unique_counts[unique_counts <= 1].index.tolist()

    # Also remove columns that are constant zero (std == 0) for numeric
    numeric = df.select_dtypes(include=[np.number])
    std_zero = numeric.columns[numeric.std(ddof=0) == 0].tolist()

    # union
    to_drop = sorted(set(zero_var_cols + std_zero))

    # ensure MPS is dropped as requested
    if 'MPS' in df.columns and 'MPS' not in to_drop:
        to_drop.append('MPS')

    # avoid accidentally dropping MeasuredCycleDuration
    if 'MeasuredCycleDuration' in to_drop:
        to_drop.remove('MeasuredCycleDuration')

    df_pruned = df.drop(columns=to_drop, errors='ignore')

    out_csv = OUT / 'ham_veri_mould_5001_pruned.csv'
    df_pruned.to_csv(out_csv, index=False)

    summary = {
        'orig_shape': orig_shape,
        'pruned_shape': df_pruned.shape,
        'dropped_columns': to_drop,
        'pruned_csv': str(out_csv)
    }
    out_json = OUT / 'prune_model_mould_5001_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote pruned CSV: {out_csv} (rows={df_pruned.shape[0]}, cols={df_pruned.shape[1]})")
    print(f"Dropped columns: {to_drop}")


if __name__ == '__main__':
    main()
