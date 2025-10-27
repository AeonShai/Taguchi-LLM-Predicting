"""Detect and remove outliers based on distance to cluster centroids in PCA space.

Outputs:
- outputs/ham_veri_mould_5001_outliers.csv
- outputs/ham_veri_mould_5001_no_outliers_pruned.csv
- outputs/ham_veri_mould_5001_no_outliers_pca_10.csv
- outputs/detect_remove_outliers_summary.json

Method:
- Load PCA10 and clusters outputs; align rows by order.
- For each cluster, compute Euclidean distances to cluster centroid in PCA10 space.
- Use robust threshold: median + 6*MAD when cluster size >=30, else mean + 3*std.
- Mark outliers and remove them; save cleaned datasets and summary.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PCA_IN = Path('outputs/ham_veri_mould_5001_pca_10.csv')
CLUSTERS_IN = Path('outputs/ham_veri_mould_5001_clusters.csv')
PRUNED_IN = Path('outputs/ham_veri_mould_5001_pruned.csv')
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)


def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))


def main():
    if not PCA_IN.exists() or not CLUSTERS_IN.exists() or not PRUNED_IN.exists():
        print('Missing inputs. Ensure PCA, clusters, and pruned files exist.')
        return

    pca = pd.read_csv(PCA_IN)
    clusters = pd.read_csv(CLUSTERS_IN)
    pruned = pd.read_csv(PRUNED_IN)

    # align by row order — they should correspond
    if not (len(pca) == len(clusters) == len(pruned)):
        print('Input row counts do not match; aborting')
        return

    X = pca.values
    labels = clusters['cluster'].values

    outlier_mask = np.zeros(len(X), dtype=bool)
    details = {}

    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        Xi = X[idx]
        if Xi.shape[0] == 0:
            continue
        centroid = Xi.mean(axis=0)
        dists = np.linalg.norm(Xi - centroid, axis=1)
        med = np.median(dists)
        cluster_mad = mad(dists)
        if Xi.shape[0] >= 30 and cluster_mad > 0:
            thresh = med + 6 * cluster_mad
            method = 'median+6*MAD'
        else:
            mu = dists.mean()
            sigma = dists.std()
            thresh = mu + 3 * sigma
            method = 'mean+3*std'

        out_idx_local = idx[dists > thresh]
        outlier_mask[out_idx_local] = True

        details[int(lab)] = {
            'cluster_size': int(Xi.shape[0]),
            'outliers_found': int(len(out_idx_local)),
            'threshold': float(thresh),
            'method': method
        }

    outliers_df = pruned[outlier_mask].copy()
    cleaned_pruned = pruned[~outlier_mask].copy()
    cleaned_pca = pca[~outlier_mask].copy()

    outliers_csv = OUT / 'ham_veri_mould_5001_outliers.csv'
    cleaned_pruned_csv = OUT / 'ham_veri_mould_5001_no_outliers_pruned.csv'
    cleaned_pca_csv = OUT / 'ham_veri_mould_5001_no_outliers_pca_10.csv'

    outliers_df.to_csv(outliers_csv, index=False)
    cleaned_pruned.to_csv(cleaned_pruned_csv, index=False)
    cleaned_pca.to_csv(cleaned_pca_csv, index=False)

    summary = {
        'total_rows': int(len(pruned)),
        'total_outliers': int(outlier_mask.sum()),
        'per_cluster': details,
        'outliers_csv': str(outliers_csv),
        'cleaned_pruned_csv': str(cleaned_pruned_csv),
        'cleaned_pca_csv': str(cleaned_pca_csv)
    }

    with open(OUT / 'detect_remove_outliers_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Found {summary['total_outliers']} outliers. Wrote: {outliers_csv}, cleaned files saved.")


if __name__ == '__main__':
    main()
