"""Inspect the MouldCode=5001 subset and write small summaries + plots.

Produces files in `outputs/`:
- `mould_5001_missing.csv`  : column, missing_percent
- `mould_5001_uniques.csv`  : column, unique_count
- `mould_5001_numeric_cols.txt` : newline-separated numeric column names
- `mould_5001_histograms.png` : grid of histograms for numeric columns (first 24)
- `mould_5001_cat_plots.png` : barplots for selected categorical columns (low cardinality)
- prints a short console summary and suggestions.json file with columns to consider dropping/converting

Run as:
    $env:PYTHONPATH='.'; python scripts/inspect_mould_5001.py
"""
from pathlib import Path
import json
import math

import pandas as pd
import numpy as np

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def main():
    src = Path("outputs/ham_veri_mould_5001.csv")
    if not src.exists():
        print(f"Missing input file: {src}. Run the subset creation step first.")
        return

    df = pd.read_csv(src)

    # Basic console preview
    print("\n=== head() ===")
    print(df.head(5).to_string(index=False))

    print("\n=== dtypes ===")
    print(df.dtypes.value_counts())

    # Missingness
    missing = (df.isna().mean() * 100).rename("missing_percent").reset_index()
    missing.columns = ["column", "missing_percent"]
    missing.to_csv(OUT / "mould_5001_missing.csv", index=False)

    # Unique counts
    uniques = df.nunique(dropna=False).rename("unique_count").reset_index()
    uniques.columns = ["column", "unique_count"]
    uniques.to_csv(OUT / "mould_5001_uniques.csv", index=False)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    with open(OUT / "mould_5001_numeric_cols.txt", "w", encoding="utf-8") as f:
        for c in numeric_cols:
            f.write(c + "\n")

    # Pick top numeric columns for histograms (limit to 24)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    to_plot = numeric_cols[:24]
    if to_plot:
        n = len(to_plot)
        cols = 4
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten()
        for i, col in enumerate(to_plot):
            ax = axes[i]
            series = df[col].dropna()
            if series.nunique() > 50:
                ax.hist(series, bins=30, color="#3b83bd")
            else:
                series.value_counts().plot.bar(ax=ax, color="#3b83bd")
            ax.set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        fig.savefig(OUT / "mould_5001_histograms.png", dpi=150)
        plt.close(fig)

    # Categorical columns with low cardinality
    cat_candidates = []
    for col in df.columns:
        if col in numeric_cols:
            continue
        nunq = df[col].nunique(dropna=False)
        if nunq <= 50:
            cat_candidates.append((col, nunq))

    # create a simple figure for up to 6 categorical columns
    import seaborn as sns

    if cat_candidates:
        top_cats = [c for c, _ in sorted(cat_candidates, key=lambda x: x[1])][:6]
        fig, axes = plt.subplots(len(top_cats), 1, figsize=(8, 3 * len(top_cats)))
        if len(top_cats) == 1:
            axes = [axes]
        for ax, col in zip(axes, top_cats):
            vals = df[col].fillna("<NA>")
            vc = vals.value_counts().head(30)
            sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="viridis")
            ax.set_title(f"{col} (unique={df[col].nunique(dropna=False)})")
        plt.tight_layout()
        fig.savefig(OUT / "mould_5001_cat_plots.png", dpi=150)
        plt.close(fig)

    # Suggest columns to drop or convert
    suggestions = {"drop_columns": [], "convert_to_datetime": [], "notes": []}

    # heuristics: drop columns with >90% missing or high cardinality > 1000 (likely IDs)
    high_missing = missing[missing["missing_percent"] > 90]["column"].tolist()
    high_card = uniques[uniques["unique_count"] > 1000]["column"].tolist()
    suggestions["drop_columns"].extend(high_missing)
    suggestions["drop_columns"].extend(high_card)

    # detect probable date columns by name
    date_like = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "timestamp", "tarih"))]
    # but only add if dtype not numeric
    for c in date_like:
        if c not in numeric_cols:
            suggestions["convert_to_datetime"].append(c)

    # small note: keep metadata columns like CODE, MPS, MouldCode as candidates for grouping, but consider encoding
    for meta in ("CODE", "MPS", "MouldCode"):
        if meta in df.columns:
            suggestions["notes"].append(f"Column {meta}: unique={int(df[meta].nunique(dropna=False))}")

    # write suggestions
    with open(OUT / "mould_5001_suggestions.json", "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2, ensure_ascii=False)

    # print short summary
    print("\n=== summary ===")
    print(f"rows={len(df)}, cols={len(df.columns)}")
    print(f"numeric columns: {len(numeric_cols)}")
    print(f"high-missing columns (>90%): {high_missing}")
    print(f"high-cardinality columns (>1000 unique): {high_card}")
    print(f"date-like columns (by name): {date_like}")
    print("Wrote outputs to outputs/ (missing, uniques, numeric_cols, histograms, cat_plots, suggestions)")


if __name__ == "__main__":
    main()
