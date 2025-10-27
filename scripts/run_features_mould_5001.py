"""Run feature engineering for MouldCode=5001: scaling + PCA runner.

Produces outputs in `outputs/` with prefix `ham_veri_mould_5001`.
"""
from pathlib import Path
from src.features import scale_and_pca

IN = Path('outputs/ham_veri_mould_5001_pruned.csv')
OUT_PREFIX = Path('outputs/ham_veri_mould_5001')


def main():
    if not IN.exists():
        print(f"Missing input: {IN}. Run pruning/prepare steps first.")
        return
    res = scale_and_pca(IN, OUT_PREFIX, n_components=10)
    print('Feature outputs:')
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
