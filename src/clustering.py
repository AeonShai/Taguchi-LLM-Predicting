"""Clustering helpers: KMeans search and metric computations."""
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def kmeans_search(X: np.ndarray, ks: List[int], random_state: int = 42) -> Dict[int, Dict[str, Any]]:
    """Run KMeans for each k in ks and compute metrics.

    Returns a dict mapping k -> { 'model': fitted_model, 'silhouette': float, 'calinski': float }
    """
    results = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = km.fit_predict(X)
        # silhouette requires at least 2 clusters and less than n_samples
        sil = silhouette_score(X, labels) if 1 < k < len(X) else float('nan')
        cal = calinski_harabasz_score(X, labels) if 1 < k < len(X) else float('nan')
        results[k] = {'model': km, 'silhouette': float(sil), 'calinski_harabasz': float(cal)}
    return results


def best_k_by_silhouette(results: Dict[int, Dict[str, Any]]) -> int:
    best_k = None
    best_s = -1.0
    for k, info in results.items():
        s = info.get('silhouette', float('-inf'))
        if not np.isnan(s) and s > best_s:
            best_s = s
            best_k = k
    return best_k
