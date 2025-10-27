"""Restore the original cleaned 5001 dataset and produce per-parameter plots and stats.

Actions:
- Copy `outputs/ham_veri_mould_5001_cleaned.csv` -> `outputs/ham_veri_mould_5001_restored.csv` (safe baseline)
- For each numeric parameter: save a PNG with histogram + boxplot in `outputs/parameter_plots/`
- Save numeric summary CSV `outputs/parameter_stats_5001.csv`

Run:
    $env:PYTHONPATH='.'; python scripts/restore_and_plot_5001.py
"""
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

IN = Path('outputs/ham_veri_mould_5001_cleaned.csv')
RESTORED = Path('outputs/ham_veri_mould_5001_restored.csv')
OUT_DIR = Path('outputs/parameter_plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)[:200]

def main():
    if not IN.exists():
        print(f"Missing input cleaned file: {IN}. Can't restore or plot.")
        return

    # create restored copy
    shutil.copy2(IN, RESTORED)
    print(f"Restored baseline: {RESTORED}")

    df = pd.read_csv(RESTORED)

    # numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns found: {len(num_cols)}")

    stats = []

    for col in num_cols:
        s = df[col]
        cnt = int(s.count())
        missing = int(s.isna().sum())
        pct_missing = float(missing) / len(df) * 100.0
        mean = float(s.mean()) if cnt>0 else np.nan
        median = float(s.median()) if cnt>0 else np.nan
        std = float(s.std()) if cnt>0 else np.nan
        mn = float(s.min()) if cnt>0 else np.nan
        mx = float(s.max()) if cnt>0 else np.nan
        skew = float(s.skew()) if cnt>0 else np.nan
        kurt = float(s.kurtosis()) if cnt>0 else np.nan

        stats.append({
            'column': col,
            'count': cnt,
            'missing': missing,
            'pct_missing': pct_missing,
            'mean': mean,
            'median': median,
            'std': std,
            'min': mn,
            'max': mx,
            'skew': skew,
            'kurtosis': kurt
        })

        # plots: histogram + boxplot
        fig, axes = plt.subplots(2, 1, figsize=(6,6), gridspec_kw={'height_ratios': [3,1]})
        ax_hist, ax_box = axes
        try:
            ax_hist.hist(s.dropna(), bins=50, color='#3b83bd')
        except Exception:
            ax_hist.text(0.5, 0.5, 'cannot plot', ha='center')
        ax_hist.set_title(f'{col} (n={len(s)}, missing={missing})')
        # log histogram if skewed
        if pd.api.types.is_numeric_dtype(s):
            # small jitter for boxplot if needed
            ax_box.boxplot(s.dropna().values, vert=False)
        out_png = OUT_DIR / (safe_name(col) + '.png')
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('outputs/parameter_stats_5001.csv', index=False)
    print('Wrote parameter stats: outputs/parameter_stats_5001.csv')
    print(f'Wrote plots to {OUT_DIR} (one PNG per numeric parameter)')


if __name__ == '__main__':
    main()
