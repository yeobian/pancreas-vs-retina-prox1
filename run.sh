#!/usr/bin/env bash
# run.sh — Full PROX1 analysis pipeline (run from project root)
set -euo pipefail

echo "=========================================="
echo "  PROX1 Regeneration Suppressor Analysis"
echo "=========================================="

# ── Detect python / pip (handles macOS where commands are python3/pip3) ───────
if command -v python3 &>/dev/null; then
  PY=python3
elif command -v python &>/dev/null; then
  PY=python
else
  echo "ERROR: Python not found. Install from https://www.python.org/downloads/" >&2
  exit 1
fi

if command -v pip3 &>/dev/null; then
  PIP=pip3
elif command -v pip &>/dev/null; then
  PIP=pip
else
  echo "ERROR: pip not found. Run:  $PY -m ensurepip --upgrade" >&2
  exit 1
fi

echo "  Using: $($PY --version)  |  pip: $($PIP --version | awk '{print $1,$2}')"

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies …"
$PIP install -r requirements.txt -q

# ── 2. Download data ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Downloading datasets …"
$PY src/download_data.py

# ── 3. Preprocess ─────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Preprocessing …"
$PY src/preprocess_retina.py
$PY src/preprocess_pancreas.py

# ── 4. Analyse ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running PROX1 co-expression analysis …"
$PY src/prox1_analysis.py

echo ""
echo "=========================================="
echo "  Done!  Check figures/ and results/"
echo "=========================================="
