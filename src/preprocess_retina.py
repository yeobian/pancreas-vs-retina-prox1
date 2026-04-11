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


def _load_tsv(tsv_path: Path, sample_id: str) -> "ad.AnnData | None":
    """
    Load a gzipped expression matrix TSV (tab or space-delimited).

    Handles GEO files where:
    - delimiter is whitespace rather than tab
    - leading columns are string metadata (barcode, cluster, sample name)
    - only the trailing numeric columns are expression values

    Returns None if the file has no usable expression data.
    """
    # Try tab first; fall back to generic whitespace (many GEO files use spaces)
    df = None
    for sep in ("\t", r"\s+"):
        try:
            _df = pd.read_csv(tsv_path, sep=sep, index_col=0, engine="python")
            if _df.shape[1] > 0:
                df = _df
                break
        except Exception:
            continue

    if df is None or df.shape[1] == 0:
        return None

    # Separate string (metadata) columns from numeric (expression) columns.
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols  = [c for c in df.columns if df[c].dtype != object]

    if obj_cols:
        # First string column is typically cell barcodes
        obs_names = [str(bc).strip('"') for bc in df[obj_cols[0]]]
    else:
        # No string columns — index should be barcodes (or row numbers)
        obs_names = [str(bc).strip('"') for bc in df.index]

    if not num_cols:
        return None

    X         = df[num_cols].values.astype("float32")
    var_names = [str(c).strip('"') for c in num_cols]

    # If matrix looks transposed (very few rows, many cols), flip it
    if X.shape[0] < 5 and X.shape[1] > X.shape[0] * 10:
        X         = X.T
        obs_names, var_names = var_names, obs_names

    if len(obs_names) == 0 or X.shape[0] == 0:
        return None

    adata = ad.AnnData(X=X)
    adata.obs_names = obs_names
    adata.var_names = var_names
    return _attach_meta(adata, sample_id)


def _build_10x_tmp(sid: str, source_files) -> Path:
    """Copy GSM-prefixed 10X files into a clean tmp dir that sc.read_10x_mtx expects."""
    tmp = RETINA_RAW_DIR / f"_tmp_{sid}"
    tmp.mkdir(exist_ok=True)
    for f in source_files:
        # Strip the GSM prefix (and optional long sample title) so scanpy finds
        # the standard names: matrix.mtx.gz, barcodes.tsv.gz, features/genes.tsv.gz
        stem = f.name
        for prefix in (f"{sid}_", sid):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        dest = tmp / stem
        if not dest.exists():
            shutil.copy(f, dest)
    return tmp


def load_all_samples() -> ad.AnnData:
    """
    Detect the file format inside RETINA_RAW_DIR and load all samples.

    Layouts tried in order
    ----------------------
    A) Sub-directory named exactly after the GSM accession:
         RETINA_RAW_DIR/GSM4037981/matrix.mtx.gz  barcodes.tsv.gz  features.tsv.gz

    B) Any sub-directory (e.g. named after sample title) that contains .mtx files.
       Sub-directories are matched to samples in accession order.

    C) Flat directory: per-sample files with GSM prefix and any .mtx in the name:
         RETINA_RAW_DIR/GSM4037981_matrix.mtx.gz  (or any other .mtx name)

    D) Flat directory: per-sample dense TSV expression matrix (genes × cells):
         RETINA_RAW_DIR/GSM4037981_expression.tsv.gz
       Files that load with 0 data columns (barcodes/feature lists) are skipped.
    """
    # ── Print first few files for diagnosis ───────────────────────────────────
    all_files = sorted(RETINA_RAW_DIR.iterdir())
    print(f"  Raw directory contains {len(all_files)} entries. First 6:")
    for f in all_files[:6]:
        print(f"    {f.name}")

    # ── Layout B: collect all subdirectories that contain .mtx files ──────────
    mtx_subdirs = sorted(
        d for d in RETINA_RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_tmp") and any(d.glob("*.mtx*"))
    )

    adatas = []
    sample_ids = list(SAMPLE_META.keys())

    for i, sid in enumerate(sample_ids):
        # ── Layout A: subdirectory named after the GSM accession ──
        sdir = RETINA_RAW_DIR / sid
        if sdir.exists() and any(sdir.glob("*.mtx*")):
            print(f"  {sid}  [10X subdir — GSM name]")
            adatas.append(_load_10x(sdir, sid))
            continue

        # ── Layout B: any subdirectory (matched by position to sample list) ──
        if i < len(mtx_subdirs):
            print(f"  {sid}  [10X subdir — {mtx_subdirs[i].name}]")
            adatas.append(_load_10x(mtx_subdirs[i], sid))
            continue

        # ── Layout C: flat directory, any .mtx file with this GSM prefix ──
        mtx_hits = sorted(RETINA_RAW_DIR.glob(f"{sid}*.mtx*"))
        if mtx_hits:
            print(f"  {sid}  [flat MTX — {mtx_hits[0].name}]")
            tmp = _build_10x_tmp(sid, RETINA_RAW_DIR.glob(f"{sid}*"))
            adatas.append(_load_10x(tmp, sid))
            continue

        # ── Layout D: TSV expression matrix ──
        # Explicitly exclude barcodes / features / genes list files
        SKIP_KEYWORDS = ("barcodes", "features", "genes", "annotation")
        tsv_candidates = [
            f for f in sorted(RETINA_RAW_DIR.glob(f"{sid}*.tsv*"))
            if not any(kw in f.name.lower() for kw in SKIP_KEYWORDS)
        ]
        for tsv_path in tsv_candidates:
            result = _load_tsv(tsv_path, sid)
            if result is not None:
                print(f"  {sid}  [TSV — {tsv_path.name}]")
                adatas.append(result)
                break
        else:
            print(f"  {sid}  WARNING: no usable data found — skipping.", file=sys.stderr)

    if not adatas:
        # Last-resort diagnostic
        print(
            f"\nERROR: No samples could be loaded from {RETINA_RAW_DIR}.\n"
            "Full file listing:",
            file=sys.stderr,
        )
        for f in all_files[:30]:
            print(f"  {f.name}", file=sys.stderr)
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
