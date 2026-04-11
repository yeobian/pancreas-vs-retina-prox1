"""
Shared utility functions for the PROX1 analysis pipeline.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad


def ensure_dir(path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_qc(adata: ad.AnnData, save_path: Path, title: str = "QC") -> None:
    """Save a QC violin plot (n_genes, total_counts, pct_mito)."""
    ensure_dir(save_path.parent)
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    available = [m for m in metrics if m in adata.obs.columns]
    if not available:
        return
    ax = sc.pl.violin(adata, available, jitter=0.4, multi_panel=True, show=False)
    plt.suptitle(title, y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def get_gene_expression(adata: ad.AnnData, gene: str) -> np.ndarray:
    """
    Extract a single gene's expression vector from an AnnData object.
    Returns a 1-D float array (log-normalized counts).
    """
    if gene not in adata.var_names:
        raise ValueError(
            f"Gene '{gene}' not found in dataset. "
            f"First 10 var_names: {list(adata.var_names[:10])}"
        )
    idx = adata.var_names.get_loc(gene)
    X = adata.X
    if hasattr(X, "toarray"):
        return X[:, idx].toarray().flatten().astype(float)
    return np.array(X[:, idx]).flatten().astype(float)


def compute_gene_correlation(
    adata: ad.AnnData,
    target_gene: str = "PROX1",
    method: str = "spearman",
    min_expr_fraction: float = 0.01,
) -> pd.Series:
    """
    Compute the correlation of every gene with *target_gene* across all cells.

    Parameters
    ----------
    adata              : preprocessed AnnData (log-normalized)
    target_gene        : gene to correlate against
    method             : 'spearman' or 'pearson'
    min_expr_fraction  : skip genes expressed in fewer than this fraction of cells

    Returns
    -------
    pd.Series sorted descending by correlation, excluding target_gene itself.
    """
    from scipy.stats import rankdata

    target_expr = get_gene_expression(adata, target_gene)

    X = adata.X
    X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X, dtype=float)

    # Drop very lowly expressed genes to keep runtime manageable
    expr_frac = (X_dense > 0).mean(axis=0)
    keep = expr_frac >= min_expr_fraction
    X_filt = X_dense[:, keep]
    gene_names = adata.var_names[keep]

    print(f"    Computing {method} correlations for {X_filt.shape[1]:,} genes …")

    if method == "spearman":
        t_ranked = rankdata(target_expr)
        X_ranked = np.apply_along_axis(rankdata, 0, X_filt)
        t_c = t_ranked - t_ranked.mean()
        X_c = X_ranked - X_ranked.mean(axis=0)
    else:
        t_c = target_expr - target_expr.mean()
        X_c = X_filt - X_filt.mean(axis=0)

    num = t_c @ X_c
    denom = np.sqrt((t_c ** 2).sum()) * np.sqrt((X_c ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1e-10, denom)
    corr = num / denom

    result = pd.Series(corr, index=gene_names)
    result = result.drop(labels=[target_gene], errors="ignore")
    return result.sort_values(ascending=False)
