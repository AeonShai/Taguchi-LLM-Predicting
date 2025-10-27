import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

root = 'C:/Users/Beara/OneDrive/Desktop/MCP-Filesystem/Tubitak'
input_csv = os.path.join(root, 'outputs', 'ham_veri_mould_5001_pruned_with_labels.csv')
model_dir = os.path.join(root, 'outputs', 'models_5001')
os.makedirs(model_dir, exist_ok=True)

model_out = os.path.join(model_dir, 'rf_cluster_k3.pkl')
scaler_out = os.path.join(model_dir, 'scaler_rf_cluster_k3.pkl')
summary_out = os.path.join(root, 'outputs', '5001_supervised_train_summary.json')
pred_out = os.path.join(root, 'outputs', '5001_supervised_test_predictions.csv')
conf_png = os.path.join(root, 'outputs', '5001_confusion_matrix.png')
fi_png = os.path.join(root, 'outputs', '5001_feature_importances.png')

print('Loading', input_csv)
df = pd.read_csv(input_csv)

# Target
if 'cluster_k3' in df.columns:
    y = df['cluster_k3'].astype(int)
elif 'cluster_label' in df.columns:
    # map cluster1->0 etc
    y = df['cluster_label'].str.replace('cluster','').astype(int) - 1
else:
    raise SystemExit('No cluster label column found')

# Features: pick numeric columns and drop target/anomaly/cluster columns
drop_cols = ['cluster_k3','cluster_label','anomaly_flag','anomaly_score']
cols = [c for c in df.columns if c not in drop_cols]
X = df[cols].select_dtypes(include=[np.number]).copy()

if X.shape[1] == 0:
    raise SystemExit('No numeric feature columns found to train on')

# Impute median
X = X.fillna(X.median())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train_s, y_train)

# Predict
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf = confusion_matrix(y_test, y_pred)

# Feature importances
importances = clf.feature_importances_
feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)

# Save model and scaler
joblib.dump(clf, model_out)
joblib.dump(scaler, scaler_out)

# Save test predictions with probabilities
probs = clf.predict_proba(X_test_s)
probs_df = pd.DataFrame(probs, columns=[f'prob_cluster_{i}' for i in clf.classes_], index=X_test.index)
pred_df = X_test.copy()
pred_df['y_true'] = y_test
pred_df['y_pred'] = y_pred
pred_df = pd.concat([pred_df, probs_df], axis=1)
pred_df.to_csv(pred_out, index=False)

# Save confusion matrix figure
plt.figure(figsize=(5,4))
plt.imshow(conf, cmap='Blues')
plt.title('Confusion matrix (test)')
plt.xlabel('pred')
plt.ylabel('true')
plt.colorbar()
for i in range(conf.shape[0]):
    for j in range(conf.shape[1]):
        plt.text(j, i, conf[i,j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(conf_png)
plt.close()

# Feature importance plot (top 20)
topn = min(20, len(feat_imp))
names = [n for n,_ in feat_imp[:topn]][:topn]
vals = [v for _,v in feat_imp[:topn]][:topn]
plt.figure(figsize=(8,6))
plt.barh(range(len(names))[::-1], vals[::-1])
plt.yticks(range(len(names)), names[::-1])
plt.xlabel('importance')
plt.title('Top feature importances')
plt.tight_layout()
plt.savefig(fi_png)
plt.close()

# Summary
summary = {
    'n_rows': int(len(df)),
    'n_features': int(X.shape[1]),
    'train_rows': int(len(X_train)),
    'test_rows': int(len(X_test)),
    'accuracy_test': float(acc),
    'classification_report': report,
    'feature_importances_top20': [{ 'feature': f, 'importance': float(v) } for f,v in feat_imp[:20]]
}
with open(summary_out, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Training finished')
print('Model saved to', model_out)
print('Scaler saved to', scaler_out)
print('Test accuracy:', acc)
print('Test rows saved to', pred_out)
print('Summary JSON:', summary_out)
print('Confusion matrix PNG:', conf_png)
print('Feature importances PNG:', fi_png)
