import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


OUT = 'outputs'
MODELDIR = os.path.join(OUT, 'models_5001')


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def main():
    ensure_dir(OUT)
    ensure_dir(MODELDIR)

    pruned_path = os.path.join(OUT, 'ham_veri_mould_5001_pruned.csv')
    if not os.path.exists(pruned_path):
        raise FileNotFoundError(pruned_path)

    df = pd.read_csv(pruned_path)
    # keep numeric columns only for modeling
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].copy()

    # fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(os.path.join(MODELDIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # fit PCA
    n_components = min(10, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    with open(os.path.join(MODELDIR, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # fit KMeans k=3 (as per user's request)
    k = 3
    kme = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kme.fit_predict(X_pca)
    with open(os.path.join(MODELDIR, 'kmeans_k3.pkl'), 'wb') as f:
        pickle.dump(kme, f)

    # fit IsolationForest on PCA space
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    iso.fit(X_pca)
    scores = iso.decision_function(X_pca)  # higher -> more normal
    anomaly_flag = iso.predict(X_pca)  # -1 anomaly, 1 normal
    with open(os.path.join(MODELDIR, 'isolation_forest.pkl'), 'wb') as f:
        pickle.dump(iso, f)

    # save outputs: merged dataframe with labels and anomaly scores
    out_df = df.copy()
    out_df['cluster_k3'] = labels
    out_df['anomaly_score'] = scores
    out_df['anomaly_flag'] = anomaly_flag
    merged_path = os.path.join(OUT, 'ham_veri_mould_5001_pruned_with_models.csv')
    out_df.to_csv(merged_path, index=False)

    # summary
    summary = {
        'n_rows': int(len(df)),
        'n_numeric_features': int(len(num)),
        'pca_components': int(n_components),
        'kmeans_k': int(k),
        'isolation_forest_contamination': 0.03,
        'model_dir': MODELDIR,
        'merged_output': merged_path
    }
    with open(os.path.join(OUT, 'train_ai_5001_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Training complete. Models saved to', MODELDIR)
    print('Merged results saved to', merged_path)


if __name__ == '__main__':
    main()
