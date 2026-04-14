"""
Fetch and preprocess human pancreas scRNA-seq data.

Data source: CELLxGENE Census (includes Tabula Sapiens and other curated datasets).
Data is streamed on-demand — no separate download step needed.

Steps
-----
1. Connect to CELLxGENE Census and pull all human pancreas cells
2. QC filtering
3. Normalise → log1p, HVGs, scale, PCA, UMAP, Leiden clustering
4. Save to data/pancreas/pancreas_processed.h5ad

Usage
-----
    python src/preprocess_pancreas.py
Prerequisites: pip install cellxgene-census (already in requirements.txt)
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, plot_qc

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
PANCREAS_OUT = DATA_DIR / "pancreas" / "pancreas_processed.h5ad"
FIGURES_DIR  = Path("figures") / "pancreas"

# ── QC thresholds ─────────────────────────────────────────────────────────────
MIN_GENES   = 200
MAX_GENES   = 8_000
MAX_PCT_MT  = 25
MIN_CELLS   = 3

# ── Census version ────────────────────────────────────────────────────────────
CENSUS_VERSION = "stable"

# ── Subsample cap (Census returns ~870k cells; cap keeps RAM under control) ───
MAX_CELLS = 100_000


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_pancreas() -> ad.AnnData:
    """
    Fetch human pancreas cells from CELLxGENE Census.

    The Census aggregates data from hundreds of published single-cell studies,
    including the full Tabula Sapiens (the primary source for this analysis).
    Filtering by tissue='pancreas' gives ~120 k cells (as of 2024).
    """
    try:
        import cellxgene_census
    except ImportError:
        print("ERROR: cellxgene-census not installed.", file=sys.stderr)
        print("Run:  pip install cellxgene-census", file=sys.stderr)
        sys.exit(1)

    obs_cols = [
        "cell_type",
        "tissue",
        "tissue_general",
        "assay",
        "sex",
        "development_stage",
        "donor_id",
        "dataset_id",
    ]

    print("Connecting to CELLxGENE Census …")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        print("Fetching human pancreas cells …")
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_value_filter="tissue_general == 'pancreas'",
            obs_column_names=obs_cols,
        )

    print(f"Fetched {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Subsample to keep peak RAM manageable on a laptop
    if adata.n_obs > MAX_CELLS:
        sc.pp.subsample(adata, n_obs=MAX_CELLS, random_state=42)
        print(f"Subsampled → {adata.n_obs:,} cells (MAX_CELLS={MAX_CELLS:,})")

    return adata


# ── QC ────────────────────────────────────────────────────────────────────────

def run_qc(adata: ad.AnnData) -> ad.AnnData:
    print("\n--- QC ---")

    # Gene symbols are in var["feature_name"] for Census data; promote to var_names
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).values
        adata.var_names_make_unique()

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    print(f"  Before: {adata.n_obs:,} cells | {adata.n_vars:,} genes")

    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    adata = adata[adata.obs.n_genes_by_counts <= MAX_GENES].copy()
    adata = adata[adata.obs.pct_counts_mt <= MAX_PCT_MT].copy()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

    print(f"  After:  {adata.n_obs:,} cells | {adata.n_vars:,} genes")
    return adata


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(adata: ad.AnnData) -> ad.AnnData:
    print("\n--- Preprocessing ---")

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
    print(f"  Highly variable genes selected: {adata.n_vars:,}")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=42)

    # Harmony batch correction over donor/dataset
    print("  Running Harmony batch correction (key='donor_id') …")
    sc.external.pp.harmony_integrate(adata, key="donor_id", random_state=42)

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden", flavor="igraph", n_iterations=2, directed=False, random_state=42)

    return adata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(PANCREAS_OUT.parent)

    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=100, facecolor="white")

    print("=== Pancreas Preprocessing (CELLxGENE Census / Tabula Sapiens) ===\n")

    adata = fetch_pancreas()
    adata = run_qc(adata)

    print("\nSaving QC figure …")
    plot_qc(adata, FIGURES_DIR / "qc_violin.png", title="Pancreas QC")

    adata = preprocess(adata)

    print(f"\nSaving → {PANCREAS_OUT}")
    adata.write_h5ad(PANCREAS_OUT)
    print("Done!\n")
    print("Next: python src/prox1_analysis.py")


if __name__ == "__main__":
    main()
