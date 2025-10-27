"""Feature engineering helpers: scaling and PCA.

Functions:
- scale_and_pca(input_csv, out_prefix, n_components=10, random_state=42)

Outputs:
- {out_prefix}_scaled.csv
- {out_prefix}_pca_{n_components}.csv
- {out_prefix}_pca_2d.png
- {out_prefix}_pca_explained_variance.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def scale_and_pca(input_csv: str | Path, out_prefix: str | Path, n_components: int = 10, random_state: int = 42):
    input_csv = Path(input_csv)
    out_prefix = Path(out_prefix)
    df = pd.read_csv(input_csv)

    # select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'MeasuredCycleDuration' not in numeric_cols:
        raise ValueError('MeasuredCycleDuration must be present and numeric')

    X = df[numeric_cols].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
    scaled_out = out_prefix.with_suffix('')
    scaled_csv = scaled_out.parent / (scaled_out.name + '_scaled.csv')
    scaled_df.to_csv(scaled_csv, index=False)

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    pca_csv = scaled_out.parent / (scaled_out.name + f'_pca_{n_components}.csv')
    pca_df.to_csv(pca_csv, index=False)

    # 2D plot (first two components)
    if X_pca.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(X_pca[:,0], X_pca[:,1], s=8, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA 2D projection')
        png_path = scaled_out.parent / (scaled_out.name + '_pca_2d.png')
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    else:
        png_path = None

    # explained variance
    explained = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'explained_variance_ratio_cumsum': pca.explained_variance_ratio_.cumsum().tolist()
    }
    ev_json = scaled_out.parent / (scaled_out.name + '_pca_explained_variance.json')
    with open(ev_json, 'w', encoding='utf-8') as f:
        json.dump(explained, f, indent=2)

    return {
        'scaled_csv': str(scaled_csv),
        'pca_csv': str(pca_csv),
        'pca_2d_png': str(png_path) if png_path is not None else None,
        'explained_json': str(ev_json)
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python -m src.features <input_csv> <out_prefix> [n_components]')
    else:
        inp = sys.argv[1]
        outp = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        res = scale_and_pca(inp, outp, n_components=n)
        print('Wrote:', res)
