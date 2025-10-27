import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


OUT = 'outputs'


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def main():
    ensure_dir(OUT)
    pca_path = os.path.join(OUT, 'ham_veri_mould_5001_pca_10.csv')
    pruned_path = os.path.join(OUT, 'ham_veri_mould_5001_pruned.csv')

    if not os.path.exists(pca_path):
        raise FileNotFoundError(pca_path)
    if not os.path.exists(pruned_path):
        print('Warning: pruned CSV not found; will only save PCA-level clusters')

    df_pca = pd.read_csv(pca_path)

    # detect PCA numeric columns (PC1..PC10)
    pca_cols = [c for c in df_pca.columns if str(c).upper().startswith('PC')]
    if len(pca_cols) == 0:
        # fallback: take first 10 numeric columns
        pca_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()[:10]

    X = df_pca[pca_cols].values

    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    df_pca['cluster_k3'] = labels
    pca_out = os.path.join(OUT, 'ham_veri_mould_5001_pca_k3_clusters.csv')
    df_pca.to_csv(pca_out, index=False)

    metrics = {}
    try:
        metrics['silhouette'] = float(silhouette_score(X, labels))
    except Exception as e:
        metrics['silhouette'] = None
        metrics['silhouette_error'] = str(e)
    try:
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
    except Exception as e:
        metrics['calinski_harabasz'] = None
        metrics['calinski_harabasz_error'] = str(e)

    metrics['n_clusters'] = k
    metrics['pca_cols_used'] = pca_cols
    metrics_path = os.path.join(OUT, 'ham_veri_mould_5001_k3_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # scatter plot PC1 vs PC2 colored by cluster
    fig_path = os.path.join(OUT, 'ham_veri_mould_5001_pca_k3_scatter.png')
    if 'PC1' in df_pca.columns and 'PC2' in df_pca.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='cluster_k3', palette='tab10', s=30)
        plt.title('PCA 2D scatter (k=3)')
        plt.legend(title='cluster_k3')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    else:
        # save a simple pairplot of first two PCA columns available
        cols = pca_cols[:2]
        if len(cols) == 2:
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df_pca[cols[0]], y=df_pca[cols[1]], hue=labels, palette='tab10', s=30)
            plt.title('PCA scatter (k=3)')
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

    # merge labels into pruned data if available and lengths match
    if os.path.exists(pruned_path):
        df_pruned = pd.read_csv(pruned_path)
        if len(df_pruned) == len(df_pca):
            df_pruned['cluster_k3'] = labels
            merged_path = os.path.join(OUT, 'ham_veri_mould_5001_pruned_with_k3.csv')
            df_pruned.to_csv(merged_path, index=False)
        else:
            print('Warning: pruned and pca row counts differ; not merging labels.')

    print('KMeans k=3 complete. Saved:', pca_out)
    print('Metrics saved:', metrics_path)
    print('Scatter saved:', fig_path)


if __name__ == '__main__':
    main()
