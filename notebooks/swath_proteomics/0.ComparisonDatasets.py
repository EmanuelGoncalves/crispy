# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from matplotlib_venn import venn3, venn3_circles
from crispy.DataImporter import Proteomics, GeneExpression, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Cancer genes
#

cgenes = pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")


# Samplesheet
#

ss = Sample().samplesheet


# Proteomics
#

prot_gdsc = Proteomics().filter(perc_measures=None)
LOG.info(f"Proteomics GDSC: {prot_gdsc.shape}")

prot_ccle = Proteomics().broad
LOG.info(f"Proteomics CCLE: {prot_ccle.shape}")


# Gene expression
#

gexp = GeneExpression().filter()
LOG.info(f"Transcriptomics: {gexp.shape}")


# Overlaps
#

samples = list(set.intersection(set(prot_gdsc), set(prot_ccle), set(gexp)))
genes = list(
    set.intersection(set(prot_gdsc.index), set(prot_ccle.index), set(gexp.index))
)
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Venn
#

subsets = [set(ss.index), set(prot_gdsc), set(prot_ccle)]
set_labels = ["CellModelPassports", "GDSC proteomics", "CCLE Proteomics"]
set_colors = sns.color_palette("pastel").as_hex()
set_colors = [set_colors[i] for i in [0, 2, 3]]

_, ax = plt.subplots(1, 1, figsize=(2.0, 2.0))
v = venn3(
    subsets=subsets, set_labels=set_labels, set_colors=set_colors, alpha=0.8, ax=ax
)
for text in v.set_labels:
    text.set_fontsize(6)
for text in v.subset_labels:
    if text is not None:
        text.set_fontsize(5)
ax.set_title("Samples overlap")
plt.savefig(f"{RPATH}/0.Comparison_venn.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


# Cumulative sum of number of measurements per protein
#

plot_df = prot_ccle.count(1).value_counts().rename("n_proteins")
plot_df = plot_df.reset_index().rename(columns=dict(index="n_measurements"))
plot_df = pd.DataFrame(
    [
        dict(
            perc=np.round(v, 1),
            n_samples=int(np.round(v * prot_ccle.shape[1], 0)),
            n_proteins=plot_df.query(f"n_measurements >= {v * prot_ccle.shape[1]}")[
                "n_proteins"
            ].sum(),
        )
        for v in np.arange(0, 1.1, 0.1)
    ]
)

_, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=600)

ax.bar(
    plot_df["perc"],
    plot_df["n_proteins"],
    width=0.1,
    linewidth=0.3,
    color=CrispyPlot.PAL_DBGD[0],
)

for i, row in plot_df.iterrows():
    plt.text(
        row["perc"],
        row["n_proteins"] * 0.99,
        ">0" if row["n_samples"] == 0 else str(int(row["n_samples"])),
        va="top",
        ha="center",
        fontsize=5,
        zorder=10,
        rotation=90,
        color="white",
    )

ax.set_xlabel("Percentage of cell lines")
ax.set_ylabel("Number of proteins")
ax.set_title("Quantified proteins\nin CCLE Proteomics")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{RPATH}/0.Comparison_protein_measurements_cumsum_CCLE.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Correlation with Broad samples
#


def sample_corr(s1, s2):
    m = pd.concat(
        [prot_gdsc[s1].rename("sanger"), prot_ccle[s2].rename("broad")],
        axis=1,
        sort=False,
    ).dropna()
    r, p = pearsonr(m["sanger"], m["broad"])
    return dict(
        sample1=s1, sample2=s2, corr=r, pval=p, same="Yes" if s1 == s2 else "No"
    )


rand_pairs = list(
    zip(
        *(
            list(prot_gdsc[samples].sample(200, axis=1, replace=True)),
            list(prot_gdsc[samples].sample(200, axis=1, replace=True)),
        )
    )
)

sample_corr = pd.concat(
    [
        pd.DataFrame([sample_corr(s, s) for s in samples]),
        pd.DataFrame([sample_corr(s1, s2) for s1, s2 in rand_pairs if s1 != s2]),
    ]
)

fig, ax = plt.subplots(1, 1, figsize=(1, 2), dpi=600)
sns.boxplot(
    sample_corr["same"],
    sample_corr["corr"],
    ax=ax,
    order=["Yes", "No"],
    notch=True,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=CrispyPlot.PAL_YES_NO,
    showcaps=False,
    saturation=1.0,
)
ax.set_xlabel("Same cell line")
ax.set_ylabel("Pearson's R")
ax.set_title("Agreement with Broad")
ax.grid(True, axis="y", ls="-", lw=0.1, alpha=1.0, zorder=0)
plt.savefig(
    f"{RPATH}/0.Comparison_samples_corr.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Cancer genes venn
#

subsets = [set(cgenes["gene_symbol"]), set(prot_gdsc.index), set(prot_ccle.index)]
set_labels = ["Cancer genes", "GDSC proteomics", "CCLE Proteomics"]
set_colors = sns.color_palette("pastel").as_hex()
set_colors = [set_colors[i] for i in [0, 2, 3]]

_, ax = plt.subplots(1, 1, figsize=(2.0, 2.0))
v = venn3(
    subsets=subsets, set_labels=set_labels, set_colors=set_colors, alpha=0.8, ax=ax
)
for text in v.set_labels:
    text.set_fontsize(6)
for text in v.subset_labels:
    if text is not None:
        text.set_fontsize(5)
ax.set_title("Overlap with cancer gene list")
plt.savefig(
    f"{RPATH}/0.Comparison_venn_cgenes.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Cancer genes correlation with gene expression
#

def cgenes_corr(g):
    LOG.info(f"Cancer genes correlation: {g}")
    m = pd.concat([
        prot_gdsc.loc[g, samples].rename("GDSC"),
        prot_ccle.loc[g, samples].rename("CCLE"),
        gexp.loc[g, samples].rename("gexp"),
    ], axis=1, sort=False).dropna()
    gdsc_r, gdsc_p = pearsonr(m["GDSC"], m["gexp"])
    ccle_r, ccle_p = pearsonr(m["CCLE"], m["gexp"])
    return dict(
        gene=g, gdsc_corr=gdsc_r, gdsc_pval=gdsc_p, ccle_corr=ccle_r, ccle_pval=ccle_p, len=m.shape[0]
    )


cgenes_corr_df = pd.DataFrame([cgenes_corr(g) for g in cgenes["gene_symbol"] if g in genes])
cgenes_corr_df = cgenes_corr_df.query("len > 10")

_, ax = plt.subplots(1, 1, figsize=(2.0, 2.0))
ax.scatter(
    cgenes_corr_df["gdsc_corr"],
    cgenes_corr_df["ccle_corr"],
    c=CrispyPlot.PAL_DTRACE[2],
    s=10,
    lw=.3,
    edgecolor="white",
    zorder=3,
)

ax.set(xlim=(-0.1, 1), ylim=(-0.1, 1))
ax.plot([0, 1], [0, 1], transform=ax.transAxes, c=CrispyPlot.PAL_DTRACE[0], lw=.3, zorder=0)

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

n_idx = list(cgenes_corr_df.eval("gdsc_corr - ccle_corr").sort_values().head(10).index)
texts = [
    plt.text(x, y, g, color="k", fontsize=5)
    for i, (x, y, g) in cgenes_corr_df.loc[n_idx, ["gdsc_corr", "ccle_corr", "gene"]].iterrows()
]
adjust_text(
    texts, arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3)
)

ax.set_xlabel("GDSC (Pearson's R)")
ax.set_ylabel("CCLE (Pearson's R)")
ax.set_title("Cancer genes\nProtein ~ Transcript")

plt.savefig(
    f"{RPATH}/0.Comparison_cgenes_corr_scatter.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")
