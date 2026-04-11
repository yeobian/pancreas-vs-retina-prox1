#!/usr/bin/env bash
# run.sh — Full PROX1 analysis pipeline (run from project root)
set -euo pipefail

echo "=========================================="
echo "  PROX1 Regeneration Suppressor Analysis"
echo "=========================================="

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies …"
pip install -r requirements.txt -q

# ── 2. Download data ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Downloading datasets …"
python src/download_data.py

# ── 3. Preprocess ─────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Preprocessing …"
python src/preprocess_retina.py
python src/preprocess_pancreas.py

# ── 4. Analyse ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running PROX1 co-expression analysis …"
python src/prox1_analysis.py

echo ""
echo "=========================================="
echo "  Done!  Check figures/ and results/"
echo "=========================================="
