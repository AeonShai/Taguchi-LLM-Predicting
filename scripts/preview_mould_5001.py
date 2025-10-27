"""Quick preview of cleaned MouldCode=5001 data.

Prints:
- shape
- column list and dtypes
- head(10)
- describe() for numeric columns (top stats)
- top values for selected categorical columns

Run:
    $env:PYTHONPATH='.'; python scripts/preview_mould_5001.py
"""
from pathlib import Path
import pandas as pd

IN = Path('outputs/ham_veri_mould_5001_cleaned.csv')

def main():
    if not IN.exists():
        print(f"Missing cleaned file: {IN}. Run the cleaning step first.")
        return
    df = pd.read_csv(IN)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 60)

    print('\n=== shape ===')
    print(df.shape)

    print('\n=== columns (name : dtype) ===')
    for c, t in df.dtypes.items():
        print(f"{c} : {t}")

    print('\n=== head (10 rows) ===')
    print(df.head(10).to_string(index=False))

    print('\n=== numeric describe (first 20 cols) ===')
    num = df.select_dtypes(include=['number']).columns.tolist()
    print(df[num[:20]].describe().T[['count','mean','std','min','25%','50%','75%','max']])

    print('\n=== sample categorical value counts ===')
    for col in ('CODE','MPS','MouldName','OPERATOR','MouldCode'):
        if col in df.columns:
            print(f"\n-- {col} (top 10) --")
            print(df[col].fillna('<NA>').value_counts().head(10).to_string())

if __name__ == '__main__':
    main()
