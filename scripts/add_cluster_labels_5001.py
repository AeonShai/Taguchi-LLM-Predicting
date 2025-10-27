import pandas as pd

root = 'C:/Users/Beara/OneDrive/Desktop/MCP-Filesystem/Tubitak'
input_csv = root + '/outputs/ham_veri_mould_5001_pruned_with_models.csv'
out_csv = root + '/outputs/ham_veri_mould_5001_pruned_with_labels.csv'

print('Loading', input_csv)
df = pd.read_csv(input_csv)
if 'cluster_k3' not in df.columns:
    raise SystemExit('cluster_k3 column not found in input CSV')

# create human-friendly label: cluster1, cluster2, ... (1-based)
df['cluster_label'] = df['cluster_k3'].apply(lambda x: f'cluster{int(x)+1}')

# one-hot columns for each cluster
clusters = sorted(df['cluster_k3'].unique())
for c in clusters:
    col = f'cluster{int(c)+1}_flag'
    df[col] = (df['cluster_k3'] == c).astype(int)

# Save
df.to_csv(out_csv, index=False)

# Print summary
print('Saved with labels to', out_csv)
print('Total rows:', len(df))
print('Cluster label counts:')
print(df['cluster_label'].value_counts().to_dict())
print('One-hot columns added:', [f'cluster{int(c)+1}_flag' for c in clusters])
