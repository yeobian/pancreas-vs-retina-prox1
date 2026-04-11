"""
PROX1 Co-expression Analysis — Retina vs. Pancreas
====================================================

Main analysis script. Produces:
  figures/comparison/prox1_umap.png          — PROX1 expression on UMAP
  figures/comparison/prox1_violin.png        — PROX1 by cell type
  figures/comparison/top_coexpressed.png     — top co-expressed genes
  figures/comparison/venn_overlap.png        — shared vs tissue-specific genes
  figures/retina/enrichment.png              — retina pathway enrichment
  figures/pancreas/enrichment.png            — pancreas pathway enrichment
  figures/comparison/enrichment_shared.png   — shared pathway enrichment
  results/retina_prox1_coexpression.tsv      — full Spearman ρ table (retina)
  results/pancreas_prox1_coexpression.tsv    — full Spearman ρ table (pancreas)
  results/gene_comparison.tsv                — shared / tissue-specific gene lists
  results/enrichment_*.tsv                   — gProfiler enrichment tables

Usage
-----
    python src/prox1_analysis.py
Prerequisites: run preprocess_retina.py and preprocess_pancreas.py first.
"""

import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from gprofiler import GProfiler

sys.path.insert(0, str(Path(__file__).parent))
from utils import ensure_dir, get_gene_expression, compute_gene_correlation

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RETINA_H5AD   = DATA_DIR / "retina"   / "retina_processed.h5ad"
PANCREAS_H5AD = DATA_DIR / "pancreas" / "pancreas_processed.h5ad"

TARGET_GENE = "PROX1"
TOP_N       = 50       # genes to carry forward for enrichment / overlap


# ── Loading ───────────────────────────────────────────────────────────────────

def load_data() -> tuple[ad.AnnData, ad.AnnData]:
    for path, label in [(RETINA_H5AD, "retina"), (PANCREAS_H5AD, "pancreas")]:
        if not path.exists():
            print(f"ERROR: {label} H5AD not found at {path}", file=sys.stderr)
            print(f"Run:  python src/preprocess_{label}.py", file=sys.stderr)
            sys.exit(1)

    print("Loading retina …")
    retina = sc.read_h5ad(RETINA_H5AD)
    print(f"  {retina.n_obs:,} cells × {retina.n_vars:,} genes")

    print("Loading pancreas …")
    pancreas = sc.read_h5ad(PANCREAS_H5AD)
    print(f"  {pancreas.n_obs:,} cells × {pancreas.n_vars:,} genes")

    return retina, pancreas


# ── PROX1 presence ────────────────────────────────────────────────────────────

def report_prox1(adata: ad.AnnData, tissue: str) -> None:
    if TARGET_GENE not in adata.var_names:
        print(f"  {tissue}: {TARGET_GENE} NOT FOUND in gene list.")
        return
    expr = get_gene_expression(adata, TARGET_GENE)
    pct  = (expr > 0).mean() * 100
    mean = expr[expr > 0].mean() if pct > 0 else 0.0
    print(f"  {tissue}: {TARGET_GENE} in {pct:.1f}% of cells | "
          f"mean expr (positive cells) = {mean:.3f}")


# ── Figures: UMAP ─────────────────────────────────────────────────────────────

def plot_umap(retina: ad.AnnData, pancreas: ad.AnnData, out: Path) -> None:
    """Side-by-side UMAPs coloured by PROX1 expression."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, adata, title in zip(axes, [retina, pancreas], ["Retina", "Pancreas"]):
        if TARGET_GENE in adata.var_names and "X_umap" in adata.obsm:
            sc.pl.umap(
                adata, color=TARGET_GENE, ax=ax, show=False,
                title=f"{title}: {TARGET_GENE}", colorbar_loc="right",
            )
        else:
            ax.text(0.5, 0.5, f"{TARGET_GENE}\nnot available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ── Figures: violin by cell type ──────────────────────────────────────────────

def _cell_type_col(adata: ad.AnnData) -> str:
    """Pick the best available cell-type annotation column."""
    for col in ("cell_type", "celltype", "Cell_type", "leiden", "cluster"):
        if col in adata.obs.columns:
            return col
    return adata.obs.columns[0]


def plot_violin(retina: ad.AnnData, pancreas: ad.AnnData, out: Path) -> None:
    """Violin of PROX1 expression by cell type, both tissues side-by-side."""
    records = []
    for adata, tissue in [(retina, "Retina"), (pancreas, "Pancreas")]:
        if TARGET_GENE not in adata.var_names:
            continue
        expr = get_gene_expression(adata, TARGET_GENE)
        ct   = adata.obs[_cell_type_col(adata)].astype(str).values
        for e, c in zip(expr, ct):
            records.append({"tissue": tissue, "cell_type": c, TARGET_GENE: e})

    if not records:
        return

    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, tissue in zip(axes, ["Retina", "Pancreas"]):
        sub = df[df["tissue"] == tissue]
        if sub.empty:
            ax.set_title(f"{tissue}: no data")
            continue
        order = (
            sub.groupby("cell_type")[TARGET_GENE]
            .mean().sort_values(ascending=False).index
        )
        sns.violinplot(
            data=sub, x="cell_type", y=TARGET_GENE,
            order=order, ax=ax, palette="Set2",
            scale="width", inner="quartile",
        )
        ax.set_title(f"{tissue}: {TARGET_GENE} by cell type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ── Co-expression ─────────────────────────────────────────────────────────────

def compute_coexpr(adata: ad.AnnData, tissue: str) -> pd.Series:
    if TARGET_GENE not in adata.var_names:
        print(f"  {tissue}: {TARGET_GENE} not in gene list — skipping co-expression.")
        return pd.Series(dtype=float)
    print(f"\n{tissue}: computing Spearman ρ with {TARGET_GENE} …")
    return compute_gene_correlation(adata, TARGET_GENE, method="spearman")


# ── Figures: top co-expressed ─────────────────────────────────────────────────

def plot_top_coexpr(
    r_corr: pd.Series, p_corr: pd.Series, out: Path, n: int = 30
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, corr, title in zip(axes, [r_corr, p_corr], ["Retina", "Pancreas"]):
        if corr.empty:
            ax.set_title(f"{title}: no data")
            continue
        top    = corr.head(n)
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in top.values]
        ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=9)
        ax.set_xlabel(f"Spearman ρ with {TARGET_GENE}")
        ax.set_title(f"{title}: top {n} {TARGET_GENE} co-expressed genes")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ── Overlap ───────────────────────────────────────────────────────────────────

def overlap(r_corr: pd.Series, p_corr: pd.Series) -> dict:
    r_top = set(r_corr.head(TOP_N).index)
    p_top = set(p_corr.head(TOP_N).index)
    return {
        "shared":          sorted(r_top & p_top),
        "retina_unique":   sorted(r_top - p_top),
        "pancreas_unique": sorted(p_top - r_top),
    }


def plot_overlap(comp: dict, out: Path) -> None:
    cats   = ["Shared", "Retina only", "Pancreas only"]
    counts = [len(comp["shared"]), len(comp["retina_unique"]), len(comp["pancreas_unique"])]
    colors = ["#8e44ad", "#c0392b", "#2980b9"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cats, counts, color=colors, edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3, str(cnt),
            ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylabel(f"# of top-{TOP_N} {TARGET_GENE} co-expressed genes")
    ax.set_title(f"Overlap of {TARGET_GENE} co-expression programs\n(Retina vs. Pancreas)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ── gProfiler enrichment ──────────────────────────────────────────────────────

def run_enrichment(genes: list[str], label: str) -> pd.DataFrame:
    if not genes:
        return pd.DataFrame()
    gp      = GProfiler(return_dataframe=True)
    results = gp.profile(
        organism="hsapiens",
        query=genes,
        sources=["GO:BP", "GO:MF", "KEGG", "REAC"],
        significance_threshold_method="fdr",
        user_threshold=0.05,
        no_evidences=False,
    )
    if results.empty:
        print(f"  {label}: no significant terms (FDR < 0.05)")
    else:
        print(f"  {label}: {len(results)} significant terms")
    return results


def plot_enrichment(df: pd.DataFrame, out: Path, title: str, n: int = 15) -> None:
    if df.empty:
        return
    top = df.nsmallest(n, "p_value")[["name", "p_value"]].copy()
    top["-log10p"] = -np.log10(top["p_value"].clip(lower=1e-30))
    top = top.sort_values("-log10p")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top)), top["-log10p"], color="#27ae60", edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["name"], fontsize=9)
    ax.set_xlabel("−log₁₀(p-value)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(
    r_corr: pd.Series, p_corr: pd.Series, comp: dict,
    e_ret: pd.DataFrame, e_pan: pd.DataFrame, e_shared: pd.DataFrame,
) -> None:
    ensure_dir(RESULTS_DIR)

    if not r_corr.empty:
        r_corr.rename("spearman_rho").to_csv(
            RESULTS_DIR / "retina_prox1_coexpression.tsv", sep="\t", header=True
        )
    if not p_corr.empty:
        p_corr.rename("spearman_rho").to_csv(
            RESULTS_DIR / "pancreas_prox1_coexpression.tsv", sep="\t", header=True
        )

    rows = (
        [("shared",          g) for g in comp.get("shared", [])]
        + [("retina_unique",   g) for g in comp.get("retina_unique", [])]
        + [("pancreas_unique", g) for g in comp.get("pancreas_unique", [])]
    )
    if rows:
        pd.DataFrame(rows, columns=["category", "gene"]).to_csv(
            RESULTS_DIR / "gene_comparison.tsv", sep="\t", index=False
        )

    for df, name in [
        (e_ret,    "enrichment_retina"),
        (e_pan,    "enrichment_pancreas"),
        (e_shared, "enrichment_shared"),
    ]:
        if not df.empty:
            df.to_csv(RESULTS_DIR / f"{name}.tsv", sep="\t", index=False)

    print(f"  Results written to {RESULTS_DIR}/")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(r_corr: pd.Series, p_corr: pd.Series, comp: dict) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for corr, tissue in [(r_corr, "Retina"), (p_corr, "Pancreas")]:
        if not corr.empty:
            print(f"\nTop 10 {TARGET_GENE} co-expressed genes — {tissue}:")
            for gene, rho in corr.head(10).items():
                print(f"  {gene:<22} ρ = {rho:+.4f}")

    if comp.get("shared"):
        print(f"\nShared between tissues (top {TOP_N} each):")
        for g in comp["shared"][:20]:
            print(f"  {g}")
        extra = len(comp["shared"]) - 20
        if extra > 0:
            print(f"  … and {extra} more")

    print("\nFull results in results/  |  Figures in figures/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for subdir in ("retina", "pancreas", "comparison"):
        ensure_dir(FIGURES_DIR / subdir)

    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=150, facecolor="white")

    print("=" * 60)
    print(f"PROX1 REGENERATION SUPPRESSOR ANALYSIS")
    print("Retina (GSE135922) vs. Pancreas (Tabula Sapiens)")
    print("=" * 60)

    # 1. Load data
    retina, pancreas = load_data()

    # 2. PROX1 presence
    print(f"\n--- {TARGET_GENE} Expression ---")
    report_prox1(retina,   "Retina")
    report_prox1(pancreas, "Pancreas")

    # 3. UMAP & violin
    print("\n--- Generating expression plots ---")
    plot_umap(retina, pancreas,   FIGURES_DIR / "comparison" / "prox1_umap.png")
    plot_violin(retina, pancreas, FIGURES_DIR / "comparison" / "prox1_violin.png")

    # 4. Co-expression
    print("\n--- Co-expression Analysis ---")
    r_corr = compute_coexpr(retina,   "Retina")
    p_corr = compute_coexpr(pancreas, "Pancreas")

    # 5. Top genes
    print("\n--- Generating co-expression plots ---")
    plot_top_coexpr(r_corr, p_corr, FIGURES_DIR / "comparison" / "top_coexpressed.png")

    # 6. Overlap
    comp = {}
    if not r_corr.empty and not p_corr.empty:
        comp = overlap(r_corr, p_corr)
        plot_overlap(comp, FIGURES_DIR / "comparison" / "venn_overlap.png")

    # 7. Enrichment
    print("\n--- Pathway Enrichment (gProfiler) ---")
    e_ret    = run_enrichment(list(r_corr.head(TOP_N).index),      "Retina top genes")
    e_pan    = run_enrichment(list(p_corr.head(TOP_N).index),      "Pancreas top genes")
    e_shared = run_enrichment(comp.get("shared", []),              "Shared genes")

    plot_enrichment(e_ret,    FIGURES_DIR / "retina"      / "enrichment.png",
                    f"Retina: {TARGET_GENE} co-expression pathways")
    plot_enrichment(e_pan,    FIGURES_DIR / "pancreas"    / "enrichment.png",
                    f"Pancreas: {TARGET_GENE} co-expression pathways")
    plot_enrichment(e_shared, FIGURES_DIR / "comparison"  / "enrichment_shared.png",
                    f"Shared {TARGET_GENE} co-expression pathways")

    # 8. Save
    print("\n--- Saving Results ---")
    save_results(r_corr, p_corr, comp, e_ret, e_pan, e_shared)

    # 9. Summary
    print_summary(r_corr, p_corr, comp)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
