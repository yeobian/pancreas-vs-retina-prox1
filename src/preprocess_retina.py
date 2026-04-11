"""
Preprocess GSE135922 — Human Retina RPE/Choroid scRNA-seq.

Steps
-----
1. Load all 14 per-sample 10X count matrices (or TSV fall-back)
2. Concatenate into a single AnnData, attaching sample metadata
3. QC filtering (min/max genes, mitochondrial content)
4. Normalise → log1p, find HVGs, scale, PCA, UMAP, Leiden clustering
5. Save to data/retina/retina_processed.h5ad

Usage
-----
    python src/preprocess_retina.py
Prerequisites: run download_data.py first.
"""

import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, plot_qc

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
RETINA_RAW_DIR  = DATA_DIR / "retina" / "raw"
RETINA_OUT      = DATA_DIR / "retina" / "retina_processed.h5ad"
FIGURES_DIR     = Path("figures") / "retina"

# ── QC thresholds ─────────────────────────────────────────────────────────────
MIN_GENES   = 200
MAX_GENES   = 6_000
MAX_PCT_MT  = 20
MIN_CELLS   = 3

# ── Sample metadata (donors 1-3: unenriched 8-mm; donors 4-7: CD31-enriched 12-mm)
SAMPLE_META = {
    "GSM4037981": {"region": "macula",     "donor": "D1", "cd31_enriched": False, "AMD": False},
    "GSM4037982": {"region": "macula",     "donor": "D2", "cd31_enriched": False, "AMD": False},
    "GSM4037983": {"region": "macula",     "donor": "D3", "cd31_enriched": False, "AMD": True},
    "GSM4037984": {"region": "peripheral", "donor": "D1", "cd31_enriched": False, "AMD": False},
    "GSM4037985": {"region": "peripheral", "donor": "D2", "cd31_enriched": False, "AMD": False},
    "GSM4037986": {"region": "peripheral", "donor": "D3", "cd31_enriched": False, "AMD": True},
    "GSM4037987": {"region": "macula",     "donor": "D4", "cd31_enriched": True,  "AMD": True},
    "GSM4037988": {"region": "macula",     "donor": "D5", "cd31_enriched": True,  "AMD": False},
    "GSM4037989": {"region": "macula",     "donor": "D6", "cd31_enriched": True,  "AMD": False},
    "GSM4037990": {"region": "macula",     "donor": "D7", "cd31_enriched": True,  "AMD": False},
    "GSM4037991": {"region": "peripheral", "donor": "D4", "cd31_enriched": True,  "AMD": True},
    "GSM4037992": {"region": "peripheral", "donor": "D5", "cd31_enriched": True,  "AMD": False},
    "GSM4037993": {"region": "peripheral", "donor": "D6", "cd31_enriched": True,  "AMD": False},
    "GSM4037994": {"region": "peripheral", "donor": "D7", "cd31_enriched": True,  "AMD": False},
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _attach_meta(adata: ad.AnnData, sample_id: str) -> ad.AnnData:
    """Prefix barcodes and annotate obs with sample metadata."""
    adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]
    meta = SAMPLE_META.get(sample_id, {})
    for key, val in meta.items():
        adata.obs[key] = val
    adata.obs["sample"] = sample_id
    return adata


def _load_10x(sample_dir: Path, sample_id: str) -> ad.AnnData:
    adata = sc.read_10x_mtx(sample_dir, var_names="gene_symbols", cache=True, gex_only=True)
    return _attach_meta(adata, sample_id)


def _load_tsv(tsv_path: Path, sample_id: str) -> ad.AnnData:
    """Load a gzipped gene×cell TSV (rows=genes, cols=cells)."""
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    adata = ad.AnnData(X=df.T.values.astype("float32"))
    adata.obs_names = list(df.columns)
    adata.var_names = list(df.index)
    return _attach_meta(adata, sample_id)


def load_all_samples() -> ad.AnnData:
    """
    Detect the file format inside RETINA_RAW_DIR and load all samples.

    Supported layouts
    -----------------
    A) One sub-directory per sample containing MTX + barcodes + features files
       (standard 10X CellRanger output):
         RETINA_RAW_DIR/GSM4037981/matrix.mtx.gz
                                   barcodes.tsv.gz
                                   features.tsv.gz   ← or genes.tsv.gz

    B) Flat directory with per-sample MTX triplets named with the GSM prefix:
         RETINA_RAW_DIR/GSM4037981_matrix.mtx.gz  (etc.)

    C) Per-sample TSV expression files:
         RETINA_RAW_DIR/GSM4037981_expression.tsv.gz
    """
    adatas = []

    for sid in SAMPLE_META:
        # ── Layout A: sub-directory with MTX files ──
        sdir = RETINA_RAW_DIR / sid
        if sdir.exists() and any(sdir.glob("*.mtx*")):
            print(f"  {sid}  [10X subdir]")
            adatas.append(_load_10x(sdir, sid))
            continue

        # ── Layout B: flat MTX files prefixed by GSM id ──
        mtx_hits = sorted(RETINA_RAW_DIR.glob(f"{sid}*matrix.mtx*"))
        if mtx_hits:
            print(f"  {sid}  [flat MTX]")
            tmp = RETINA_RAW_DIR / f"_tmp_{sid}"
            tmp.mkdir(exist_ok=True)
            for f in RETINA_RAW_DIR.glob(f"{sid}*"):
                # strip the GSM prefix so scanpy finds matrix.mtx.gz etc.
                stem = f.name.replace(f"{sid}_", "")
                dest = tmp / stem
                if not dest.exists():
                    shutil.copy(f, dest)
            adatas.append(_load_10x(tmp, sid))
            continue

        # ── Layout C: TSV expression files ──
        tsv_hits = (
            sorted(RETINA_RAW_DIR.glob(f"{sid}*expression*.tsv.gz"))
            or sorted(RETINA_RAW_DIR.glob(f"{sid}*.tsv.gz"))
        )
        if tsv_hits:
            print(f"  {sid}  [TSV]")
            adatas.append(_load_tsv(tsv_hits[0], sid))
            continue

        print(f"  {sid}  WARNING: no data found — skipping.", file=sys.stderr)

    if not adatas:
        print(
            f"\nERROR: No samples loaded from {RETINA_RAW_DIR}.\n"
            "Run download_data.py first and verify the tar was extracted correctly.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nLoaded {len(adatas)} / {len(SAMPLE_META)} samples — concatenating …")
    combined = ad.concat(adatas, join="outer", fill_value=0)
    combined.obs_names_make_unique()
    return combined


# ── QC ────────────────────────────────────────────────────────────────────────

def run_qc(adata: ad.AnnData) -> ad.AnnData:
    print("\n--- QC ---")
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

    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")

    return adata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(RETINA_OUT.parent)

    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=100, facecolor="white")

    print("=== Retina Preprocessing (GSE135922) ===\n")
    print("Loading samples …")
    adata = load_all_samples()

    adata = run_qc(adata)

    print("\nSaving QC figure …")
    plot_qc(adata, FIGURES_DIR / "qc_violin.png", title="Retina QC")

    adata = preprocess(adata)

    print(f"\nSaving → {RETINA_OUT}")
    adata.write_h5ad(RETINA_OUT)
    print("Done!\n")
    print("Next: python src/preprocess_pancreas.py")


if __name__ == "__main__":
    main()
