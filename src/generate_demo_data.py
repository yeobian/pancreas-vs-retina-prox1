"""
Generate synthetic demo data for PROX1 analysis pipeline.

Creates realistic (but synthetic) scRNA-seq AnnData objects for both
retina and pancreas so the full analysis can run without internet access.

Cell type compositions and PROX1 expression patterns are based on
published literature:
  - Retina: Voigt et al. 2019 (RPE/choroid atlas, GSE135922)
  - Pancreas: Tabula Sapiens (Baron et al. 2016 proportions)

Run once:
    python src/generate_demo_data.py
Then proceed straight to:
    python src/prox1_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir

# ── Reproducibility ───────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
RETINA_OUT   = DATA_DIR / "retina"   / "retina_processed.h5ad"
PANCREAS_OUT = DATA_DIR / "pancreas" / "pancreas_processed.h5ad"

# ── Shared gene universe (3 000 genes, includes key markers) ──────────────────
N_GENES = 3_000

# Biologically meaningful gene groups (used to shape expression patterns)
PROX1_COEXPR_LYMPH = [          # Lymphatic/vascular TFs that co-express with PROX1
    "LYVE1", "PDPN", "FLT4", "NRP2", "VEGFC", "FOXC2",
    "SOX18", "GATA2", "COUP-TFII", "EFNB2", "ANGPT2",
]
PROX1_COEXPR_RETINA = [         # Retina endothelial markers
    "PECAM1", "VWF", "CDH5", "TIE1", "ANGPT1", "TEK",
    "KDR", "ESAM", "CLDN5", "ICAM2",
]
PROX1_COEXPR_PANCREAS = [       # Pancreatic ductal / endocrine markers
    "SOX9", "KRT19", "HNF1B", "FOXA2", "NKX6-1", "PDX1",
    "NEUROD1", "PTF1A", "RBPJ", "HES1",
]
HOUSEKEEPING = ["ACTB", "GAPDH", "B2M", "HPRT1", "POLR2A"]

EXTRA_GENES = (
    PROX1_COEXPR_LYMPH + PROX1_COEXPR_RETINA
    + PROX1_COEXPR_PANCREAS + HOUSEKEEPING + ["PROX1"]
)
N_RANDOM = N_GENES - len(EXTRA_GENES)
RANDOM_GENES = [f"GENE{i:04d}" for i in range(N_RANDOM)]
ALL_GENES = EXTRA_GENES + RANDOM_GENES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _neg_binom_counts(n_cells: int, mean: float, dispersion: float = 2.0) -> np.ndarray:
    """Sample from negative binomial to mimic scRNA-seq counts."""
    p = dispersion / (mean + dispersion)
    return RNG.negative_binomial(dispersion, p, size=n_cells).astype("float32")


def _make_expression_matrix(
    n_cells: int,
    cell_type_labels: np.ndarray,
    cell_type_profiles: dict,   # {cell_type: {gene: mean_count}}
) -> np.ndarray:
    """Build a sparse-ish count matrix n_cells x N_GENES."""
    X = np.zeros((n_cells, N_GENES), dtype="float32")
    gene_idx = {g: i for i, g in enumerate(ALL_GENES)}

    for ct in np.unique(cell_type_labels):
        mask    = cell_type_labels == ct
        n       = mask.sum()
        profile = cell_type_profiles.get(ct, {})

        # Background expression for all genes
        for i in range(N_GENES):
            X[mask, i] = _neg_binom_counts(n, mean=0.05)

        # Cell-type-specific elevated expression
        for gene, mean_cnt in profile.items():
            if gene in gene_idx:
                j = gene_idx[gene]
                X[mask, j] = _neg_binom_counts(n, mean=mean_cnt)

    # Add some technical noise
    dropout_mask = RNG.random((n_cells, N_GENES)) < 0.7
    X[dropout_mask] = 0
    return X


def _preprocess(adata: ad.AnnData) -> ad.AnnData:
    """Run standard scanpy preprocessing on an AnnData."""
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(30, adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden", random_state=42)
    return adata


# ── Retina dataset ────────────────────────────────────────────────────────────

def make_retina(n_cells: int = 5_000) -> ad.AnnData:
    """
    Simulate GSE135922-like retina RPE/choroid data.

    Cell types (approximate fractions from Voigt 2019):
      Endothelial  35%  — PROX1+ (lymphatic subset), PECAM1+, VWF+
      RPE          25%  — RPE65+, RLBP1+
      Pericyte     12%  — PDGFRB+, ACTA2+
      Fibroblast   10%  — DCN+, COL1A1+
      Melanocyte    8%  — MITF+, DCT+
      Mast cell     5%  — KIT+, TPSAB1+
      Macrophage    5%  — CD68+, AIF1+
    """
    cell_types = RNG.choice(
        ["Endothelial", "RPE", "Pericyte", "Fibroblast",
         "Melanocyte", "Mast cell", "Macrophage"],
        size=n_cells,
        p=[0.35, 0.25, 0.12, 0.10, 0.08, 0.05, 0.05],
    )

    profiles = {
        "Endothelial": {
            "PROX1": 3.5, "PECAM1": 8.0, "VWF": 6.0, "CDH5": 7.0,
            "LYVE1": 4.0, "FLT4": 3.0, "CLDN5": 5.0,
            "PDPN": 2.5, "NRP2": 2.0, "VEGFC": 2.0, "ANGPT2": 3.0,
            "KDR": 4.0, "ICAM2": 3.5, "TIE1": 4.0, "TEK": 3.5,
            "ESAM": 3.0, "ANGPT1": 2.5, "EFNB2": 2.0, "FOXC2": 1.5,
        },
        "RPE": {
            "PROX1": 0.1, "ACTB": 6.0, "GAPDH": 6.0, "B2M": 4.0,
        },
        "Pericyte": {
            "PROX1": 0.3, "ACTB": 5.0, "ANGPT1": 2.0,
        },
        "Fibroblast": {
            "PROX1": 0.2, "ACTB": 5.5, "GAPDH": 5.0,
        },
        "Melanocyte": {
            "PROX1": 0.2, "ACTB": 5.0,
        },
        "Mast cell": {
            "PROX1": 0.1, "B2M": 3.0,
        },
        "Macrophage": {
            "PROX1": 0.2, "B2M": 4.0, "ACTB": 5.0,
        },
    }

    X = _make_expression_matrix(n_cells, cell_types, profiles)

    adata = ad.AnnData(X=X)
    adata.var_names = ALL_GENES
    adata.obs_names = [f"retina_cell_{i}" for i in range(n_cells)]
    adata.obs["cell_type"] = cell_types
    adata.obs["region"]    = RNG.choice(["macula", "peripheral"], size=n_cells, p=[0.5, 0.5])
    adata.obs["donor"]     = RNG.choice(["D1","D2","D3","D4","D5","D6","D7"], size=n_cells)
    adata.obs["AMD"]       = adata.obs["donor"].isin(["D3", "D4"]).astype(str)
    adata.obs["tissue"]    = "retina"

    return adata


# ── Pancreas dataset ──────────────────────────────────────────────────────────

def make_pancreas(n_cells: int = 6_000) -> ad.AnnData:
    """
    Simulate Tabula Sapiens–like pancreas data.

    Cell types (Baron et al. 2016 proportions):
      Acinar      40%  — AMY2A+, PRSS1+
      Ductal      25%  — KRT19+, SOX9+, PROX1+ (moderate)
      Beta        15%  — INS+, IAPP+
      Alpha        8%  — GCG+, ARX+
      Endothelial  5%  — PECAM1+, VWF+, PROX1+ (lymphatic)
      Delta        4%  — SST+
      Fibroblast   3%  — DCN+
    """
    cell_types = RNG.choice(
        ["Acinar", "Ductal", "Beta", "Alpha", "Endothelial", "Delta", "Fibroblast"],
        size=n_cells,
        p=[0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03],
    )

    profiles = {
        "Ductal": {
            "PROX1": 2.8, "KRT19": 7.0, "SOX9": 5.0, "HNF1B": 4.5,
            "FOXA2": 4.0, "RBPJ": 2.5, "HES1": 2.0, "PTF1A": 1.5,
            "NKX6-1": 2.0, "PDX1": 2.5, "NEUROD1": 1.5,
        },
        "Endothelial": {
            "PROX1": 3.0, "PECAM1": 7.0, "VWF": 5.5, "CDH5": 6.5,
            "LYVE1": 3.5, "FLT4": 2.5, "CLDN5": 4.5, "PDPN": 2.0,
            "ANGPT2": 2.5, "KDR": 3.5,
        },
        "Acinar": {
            "PROX1": 0.15, "ACTB": 6.0, "GAPDH": 6.5,
        },
        "Beta": {
            "PROX1": 0.4, "ACTB": 5.5, "GAPDH": 5.5,
        },
        "Alpha": {
            "PROX1": 0.3, "ACTB": 5.0,
        },
        "Delta": {
            "PROX1": 0.2, "ACTB": 4.5,
        },
        "Fibroblast": {
            "PROX1": 0.2, "ACTB": 5.0, "GAPDH": 5.0,
        },
    }

    X = _make_expression_matrix(n_cells, cell_types, profiles)

    adata = ad.AnnData(X=X)
    adata.var_names = ALL_GENES
    adata.obs_names = [f"pancreas_cell_{i}" for i in range(n_cells)]
    adata.obs["cell_type"]        = cell_types
    adata.obs["tissue"]           = "pancreas"
    adata.obs["tissue_general"]   = "pancreas"
    adata.obs["sex"]              = RNG.choice(["male", "female"], size=n_cells)
    adata.obs["donor_id"]         = RNG.choice([f"TSP{i}" for i in range(1, 9)], size=n_cells)
    adata.obs["development_stage"] = RNG.choice(
        ["30-year-old stage", "45-year-old stage", "60-year-old stage"], size=n_cells
    )

    return adata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(RETINA_OUT.parent)
    ensure_dir(PANCREAS_OUT.parent)

    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=80, facecolor="white")

    print("=== Generating synthetic demo data ===\n")

    print("Building retina dataset (5 000 cells × 3 000 genes) …")
    retina = make_retina(5_000)
    print("Preprocessing retina …")
    retina = _preprocess(retina)
    retina.write_h5ad(RETINA_OUT)
    print(f"  Saved → {RETINA_OUT}  ({retina.n_obs:,} cells × {retina.n_vars:,} HVGs)\n")

    print("Building pancreas dataset (6 000 cells × 3 000 genes) …")
    pancreas = make_pancreas(6_000)
    print("Preprocessing pancreas …")
    pancreas = _preprocess(pancreas)
    pancreas.write_h5ad(PANCREAS_OUT)
    print(f"  Saved → {PANCREAS_OUT}  ({pancreas.n_obs:,} cells × {pancreas.n_vars:,} HVGs)\n")

    print("Demo data ready. Run:  python src/prox1_analysis.py")


if __name__ == "__main__":
    main()
