"""
Download all datasets required for the PROX1 regeneration suppressor analysis.

Datasets
--------
1. GSE135922 — Human retina RPE/choroid scRNA-seq (NCBI GEO, 14 samples, 10X Genomics)
2. Tabula Sapiens pancreas — fetched live via CELLxGENE Census during preprocessing
   (no separate download step needed for pancreas)

Usage
-----
    python src/download_data.py
"""

import gzip
import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
RETINA_DIR  = DATA_DIR / "retina" / "raw"

# ── URLs ──────────────────────────────────────────────────────────────────────
# NCBI GEO supplementary tar (contains all 14 per-sample 10X files)
GEO_RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135922/suppl/GSE135922_RAW.tar"
)

# ── Sample metadata (from series matrix) ─────────────────────────────────────
SAMPLES = {
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, label: str) -> None:
    """Stream-download *url* to *dest* with a progress bar. Skip if present."""
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label} …")

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ERROR: could not reach {url}\n  {exc}", file=sys.stderr)
        sys.exit(1)

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        desc=label, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65_536):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"  Saved → {dest}")


# ── Step 1: retina data ───────────────────────────────────────────────────────

def download_retina() -> None:
    print("\n=== Retina Data (GSE135922) ===")

    raw_tar = DATA_DIR / "GSE135922_RAW.tar"
    _download_file(GEO_RAW_URL, raw_tar, "GSE135922_RAW.tar")

    if RETINA_DIR.exists() and any(RETINA_DIR.iterdir()):
        print(f"  [skip] already extracted to {RETINA_DIR}/")
        return

    print(f"  Extracting → {RETINA_DIR}/ …")
    RETINA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(raw_tar, "r") as tar:
        tar.extractall(path=RETINA_DIR)

    # Decompress any non-MTX .gz files (barcodes / features TSVs)
    for gz_file in RETINA_DIR.rglob("*.tsv.gz"):
        out = gz_file.with_suffix("")
        if not out.exists():
            with gzip.open(gz_file, "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)

    print("  Extraction complete.")


# ── Step 2: verify pancreas tooling ──────────────────────────────────────────

def check_pancreas_tooling() -> None:
    print("\n=== Pancreas Data (Tabula Sapiens via CELLxGENE Census) ===")
    try:
        import cellxgene_census  # noqa: F401
        print("  cellxgene-census is installed. Data is fetched live during preprocessing.")
        print("  No separate download step required.")
    except ImportError:
        print("  ERROR: cellxgene-census is not installed.", file=sys.stderr)
        print("  Run:  pip install cellxgene-census", file=sys.stderr)
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    download_retina()
    check_pancreas_tooling()

    print("\nAll done! Next steps:")
    print("  python src/preprocess_retina.py")
    print("  python src/preprocess_pancreas.py")
