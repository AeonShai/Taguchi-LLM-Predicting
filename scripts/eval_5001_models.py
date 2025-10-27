import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# Paths
root = 'C:/Users/Beara/OneDrive/Desktop/MCP-Filesystem/Tubitak'
csv_path = root + '/outputs/ham_veri_mould_5001_pruned_with_models.csv'
metrics_path = root + '/outputs/ham_veri_mould_5001_k3_metrics.json'
summary_out = root + '/outputs/5001_evaluation_summary.json'
top_out = root + '/outputs/5001_top_anomalies.csv'
hist_out = root + '/outputs/5001_anomaly_score_hist.png'

# Read data
df = pd.read_csv(csv_path)
# Read clustering metrics if available
metrics = {}
try:
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
except Exception:
    metrics = {}

n = len(df)
cluster_counts = df['cluster_k3'].value_counts().to_dict()
# anomaly_flag: -1 means anomaly (IsolationForest convention), 1 inlier
anomaly_counts = df['anomaly_flag'].value_counts().to_dict()
num_anomalies = int(anomaly_counts.get(-1, 0))
frac_anomalies = num_anomalies / n if n else 0.0

# Per-cluster anomaly counts
per_cluster = {}
for c, g in df.groupby('cluster_k3'):
    anom = int((g['anomaly_flag'] == -1).sum())
    per_cluster[int(c)] = { 'cluster_size': int(len(g)), 'anomaly_count': anom, 'anomaly_fraction': float(anom/len(g)) }

# Top anomalies (most negative/lowest anomaly_score) — sort ascending
topN = 20
if 'anomaly_score' in df.columns:
    top = df.sort_values('anomaly_score').head(topN)
else:
    top = df[df['anomaly_flag']==-1].head(topN)

# Save top anomalies
top.to_csv(top_out, index=False)

# Save histogram of anomaly_score
if 'anomaly_score' in df.columns:
    plt.figure(figsize=(6,4))
    plt.hist(df['anomaly_score'], bins=60, color='#2c7fb8')
    plt.axvline(df.loc[df['anomaly_flag']==-1,'anomaly_score'].median() if num_anomalies>0 else 0, color='red', linestyle='--', label='median flagged')
    plt.xlabel('anomaly_score')
    plt.ylabel('count')
    plt.title('Anomaly score distribution (5001 subset)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_out)

summary = {
    'total_rows': int(n),
    'cluster_counts': {int(k): int(v) for k,v in cluster_counts.items()},
    'num_anomalies': int(num_anomalies),
    'fraction_anomalies': float(frac_anomalies),
    'per_cluster_anomaly': per_cluster,
    'clustering_metrics': metrics
}

with open(summary_out, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Wrote evaluation summary:', summary_out)
print('Total rows:', n)
print('Num anomalies:', num_anomalies, 'fraction:', frac_anomalies)
print('Cluster counts:', cluster_counts)
print('Per-cluster anomaly (saved):', per_cluster)
print('Top anomalies CSV:', top_out)
print('Histogram PNG:', hist_out)
