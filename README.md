# Prox1 – Regeneration Suppressor Analysis

This project explores the gene **PROX1** as a potential global regeneration suppressor, not just in the retina (as shown in existing literature), but also in the pancreas using single-cell RNA-seq data.

## Goals
- Analyze PROX1 co-expression patterns in retina vs. pancreas
- Identify top co-expressed genes shared and unique to each tissue
- Suggest new therapeutic targets via pathway enrichment
- Compare PROX1 expression by cell type

## Data Sources
| Dataset | Tissue | Access |
|---|---|---|
| GSE135922 | Human retina RPE/choroid | NCBI GEO (auto-downloaded) |
| Tabula Sapiens | Human pancreas | CELLxGENE Census (streamed) |

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/yeobian/pancreas-vs-retina-prox1.git
cd pancreas-vs-retina-prox1

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# 3. Run the full pipeline
bash run.sh
```

Or run each step individually:

```bash
pip install -r requirements.txt
python src/download_data.py        # download GSE135922 from NCBI GEO
python src/preprocess_retina.py    # QC + cluster retina data
python src/preprocess_pancreas.py  # fetch + cluster pancreas data
python src/prox1_analysis.py       # co-expression, overlap, enrichment
```

## Project Structure

```
├── requirements.txt               # pinned Python dependencies
├── run.sh                         # one-command pipeline runner
├── src/
│   ├── download_data.py           # downloads GSE135922 from NCBI FTP
│   ├── preprocess_retina.py       # QC, normalise, cluster retina cells
│   ├── preprocess_pancreas.py     # fetch & preprocess pancreas cells
│   ├── prox1_analysis.py          # PROX1 co-expression & enrichment
│   └── utils.py                   # shared helpers
├── data/                          # created at runtime (gitignored)
│   ├── retina/
│   └── pancreas/
├── figures/                       # output plots (gitignored)
└── results/                       # output TSV tables (gitignored)
```

## Outputs

| File | Description |
|---|---|
| `figures/comparison/prox1_umap.png` | PROX1 expression on UMAP (both tissues) |
| `figures/comparison/prox1_violin.png` | PROX1 by cell type |
| `figures/comparison/top_coexpressed.png` | Top co-expressed genes |
| `figures/comparison/venn_overlap.png` | Shared vs tissue-specific genes |
| `figures/*/enrichment.png` | gProfiler pathway enrichment charts |
| `results/retina_prox1_coexpression.tsv` | Spearman ρ for all retina genes |
| `results/pancreas_prox1_coexpression.tsv` | Spearman ρ for all pancreas genes |
| `results/gene_comparison.tsv` | Shared / retina-only / pancreas-only genes |
| `results/enrichment_*.tsv` | Full gProfiler enrichment tables |

## Tools
Python · Scanpy · Pandas · Matplotlib · Seaborn · gProfiler · CELLxGENE Census

## Author
[Yeobi Hobson] — GWU
