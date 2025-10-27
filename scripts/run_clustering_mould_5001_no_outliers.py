"""Run clustering on the no-outliers PCA + pruned data.

Inputs expected:
- outputs/ham_veri_mould_5001_no_outliers_pca_10.csv
- outputs/ham_veri_mould_5001_no_outliers_pruned.csv

Outputs:
- outputs/ham_veri_mould_5001_clusters_no_outliers.csv
- outputs/ham_veri_mould_5001_cluster_metrics_no_outliers.json
- outputs/ham_veri_mould_5001_clusters_no_outliers_2d.png
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.clustering import kmeans_search, best_k_by_silhouette

PCA_IN = Path('outputs/ham_veri_mould_5001_no_outliers_pca_10.csv')
PRUNED_IN = Path('outputs/ham_veri_mould_5001_no_outliers_pruned.csv')
OUT_CLUSTER_CSV = Path('outputs/ham_veri_mould_5001_clusters_no_outliers.csv')
OUT_METRICS = Path('outputs/ham_veri_mould_5001_cluster_metrics_no_outliers.json')
OUT_PLOT = Path('outputs/ham_veri_mould_5001_clusters_no_outliers_2d.png')


def main():
    if not PCA_IN.exists() or not PRUNED_IN.exists():
        print('Missing no-outliers inputs; run detect_remove_outliers_mould_5001.py first.')
        return

    pca = pd.read_csv(PCA_IN)
    pruned = pd.read_csv(PRUNED_IN)
    X = pca.values

    ks = list(range(2, 7))
    results = kmeans_search(X, ks)
    metrics = {k: {'silhouette': info['silhouette'], 'calinski_harabasz': info['calinski_harabasz']} for k, info in results.items()}
    with open(OUT_METRICS, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    best_k = best_k_by_silhouette(results)
    if best_k is None:
        print('No valid best k found; exiting')
        return

    best_model = results[best_k]['model']
    labels = best_model.predict(X)

    merged = pruned.copy()
    merged['cluster'] = labels
    if pca.shape[1] >= 2:
        merged['PC1'] = pca.iloc[:, 0]
        merged['PC2'] = pca.iloc[:, 1]

    merged.to_csv(OUT_CLUSTER_CSV, index=False)

    if pca.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7,6))
        sc = ax.scatter(merged['PC1'], merged['PC2'], c=merged['cluster'], cmap='tab10', s=12, alpha=0.8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'KMeans clusters (k={best_k}) no outliers')
        legend1 = ax.legend(*sc.legend_elements(), title='cluster')
        ax.add_artist(legend1)
        fig.tight_layout()
        fig.savefig(OUT_PLOT, dpi=150)
        plt.close(fig)

    print(f'Wrote clusters CSV: {OUT_CLUSTER_CSV} (k={best_k})')
    print(f'Wrote metrics JSON: {OUT_METRICS}')
    print(f'Wrote 2D plot: {OUT_PLOT}')


if __name__ == '__main__':
    main()
