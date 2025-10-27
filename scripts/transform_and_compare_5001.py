import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import NearestNeighbors


OUT_DIR = os.path.join('outputs')


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def load_restored():
    path = os.path.join(OUT_DIR, 'ham_veri_mould_5001_restored.csv')
    return pd.read_csv(path)


def load_stats():
    path = os.path.join(OUT_DIR, 'parameter_stats_5001.csv')
    return pd.read_csv(path, index_col=0)


def apply_transforms(df, stats, top_n=10):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skew_series = stats['skew']
    # keep only numeric cols
    skew_series = skew_series.loc[skew_series.index.isin(numeric_cols)]
    top = skew_series.abs().sort_values(ascending=False).head(top_n).index.tolist()

    transformed = df.copy()
    transform_log_cols = []
    transform_yj_cols = []
    yj_models = {}

    for col in top:
        col_min = transformed[col].min()
        if pd.isna(col_min):
            continue
        if col_min >= 0:
            # safe to use log1p
            transformed[col] = np.log1p(transformed[col])
            transform_log_cols.append(col)
        else:
            # use Yeo-Johnson (supports negatives)
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            vals = transformed[[col]].fillna(transformed[col].median()).values
            try:
                transformed[col] = pt.fit_transform(vals).flatten()
                transform_yj_cols.append(col)
                yj_models[col] = None  # not saving model internals now
            except Exception:
                # fallback to log1p on shifted positive values
                shift = abs(transformed[col].min()) + 1e-6
                transformed[col] = np.log1p(transformed[col] + shift)
                transform_log_cols.append(col + '_shifted')

    # save transformed CSV
    out_path = os.path.join(OUT_DIR, 'ham_veri_mould_5001_transformed.csv')
    transformed.to_csv(out_path, index=False)

    # plots
    plots_dir = os.path.join(OUT_DIR, 'parameter_plots', 'transformed')
    ensure_dir(plots_dir)
    for col in top:
        plt.figure(figsize=(10,4))
        ax1 = plt.subplot(1,2,1)
        sns.histplot(df[col].dropna(), bins=40, kde=True, color='C0', ax=ax1)
        ax1.set_title(f'Original: {col}')
        ax2 = plt.subplot(1,2,2)
        sns.histplot(transformed[col].dropna(), bins=40, kde=True, color='C1', ax=ax2)
        ax2.set_title(f'Transformed: {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{col}_before_after.png'))
        plt.close()

    summary = {
        'top_transformed_by_abs_skew': top,
        'log1p_columns': transform_log_cols,
        'yeo_johnson_columns': transform_yj_cols,
        'transformed_csv': out_path,
        'plots_dir': plots_dir
    }
    with open(os.path.join(OUT_DIR, 'transform_summary_5001.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return transformed, summary


def map_outliers_to_original(df_restored):
    out_path = os.path.join(OUT_DIR, 'ham_veri_mould_5001_outliers.csv')
    if not os.path.exists(out_path):
        print('No outliers file found:', out_path)
        return None
    df_out = pd.read_csv(out_path)

    # choose numeric intersection columns for matching
    num_rest = df_restored.select_dtypes(include=[np.number]).columns.tolist()
    num_out = df_out.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in num_out if c in num_rest]
    if len(cols) == 0:
        print('No numeric intersection columns to match on')
        return None

    A = df_restored[cols].fillna(0).values
    B = df_out[cols].fillna(0).values
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(A)
    dists, idxs = nbrs.kneighbors(B)
    mapped = df_restored.iloc[idxs.flatten()].copy()
    mapped.reset_index(inplace=True)
    mapped = mapped.rename(columns={'index':'restored_index'})
    mapped['match_distance'] = dists.flatten()

    mapped_path = os.path.join(OUT_DIR, 'ham_veri_mould_5001_outliers_original_features.csv')
    mapped.to_csv(mapped_path, index=False)

    # compute z-scores of mapped rows vs population for numeric cols
    pop = df_restored[cols]
    pop_mean = pop.mean()
    pop_std = pop.std().replace(0, np.nan)
    z = (mapped[cols] - pop_mean) / pop_std

    # pick top features by mean absolute z
    mean_abs_z = z.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_z.head(20).index.tolist()

    # heatmap for these features
    heat = z[top_features]
    plt.figure(figsize=(max(6, len(top_features)*0.3), max(6, heat.shape[0]*0.25)))
    sns.heatmap(heat, cmap='RdBu_r', center=0, cbar_kws={'label':'z-score'})
    plt.title('Outliers (rows) vs population (z-scores) — top features')
    heatmap_path = os.path.join(OUT_DIR, 'outliers_vs_population_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    # summary JSON
    summary = {
        'mapped_outliers_csv': mapped_path,
        'heatmap': heatmap_path,
        'top_features_by_mean_abs_z': top_features,
        'mean_abs_z_top': mean_abs_z.head(20).to_dict()
    }
    with open(os.path.join(OUT_DIR, 'pca_outliers_vs_original_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return mapped, summary


def main():
    ensure_dir(OUT_DIR)
    df = load_restored()
    stats = load_stats()
    transformed, tsummary = apply_transforms(df, stats, top_n=10)
    mapped, msummary = map_outliers_to_original(df)
    print('Transform summary written:', tsummary.get('transformed_csv'))
    print('Outliers mapped:', msummary.get('mapped_outliers_csv') if msummary else None)


if __name__ == '__main__':
    main()
