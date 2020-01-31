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
from scipy.stats import pearsonr
from crispy.GIPlot import GIPlot
from crispy.CrispyPlot import CrispyPlot
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Protein abundance attenuation
#

p_attenuated = pd.read_csv(f"{DPATH}/protein_attenuation_table.csv", index_col=0)


# SWATH proteomics
#

prot = Proteomics().filter()
LOG.info(f"Proteomics: {prot.shape}")


# Gene expression
#

gexp = GeneExpression().filter(subset=list(prot))
LOG.info(f"Transcriptomics: {gexp.shape}")


# CRISPR
#

crispr = CRISPR().filter(subset=list(prot))
LOG.info(f"CRISPR: {crispr.shape}")


# Overlaps
#

ss = Sample().samplesheet
ss = ss[ss["tissue"] != "Haematopoietic and Lymphoid"]

samples = list(set.intersection(set(prot), set(gexp)))
genes = list(set.intersection(set(prot.index), set(gexp.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Protein ~ Transcript correlations
#

def corr_genes(g):
    mprot = prot.loc[g, samples].dropna()
    mgexp = gexp.loc[g, mprot.index]
    r, p = pearsonr(mprot, mgexp)
    return dict(gene=g, corr=r, pval=p, len=len(mprot.index))


res = pd.DataFrame([corr_genes(g) for g in genes]).sort_values("pval")
res["fdr"] = multipletests(res["pval"], method="fdr_bh")[1]
res["attenuation"] = p_attenuated.reindex(res["gene"])["attenuation_potential"].replace(np.nan, "NA").values
res.to_csv(f"{RPATH}/0.Protein_Gexp_correlations.csv.gz", index=False, compression="gzip")


# Protein ~ GExp correlation histogram
#

_, ax = plt.subplots(1, 1, figsize=(2.0, 1.5))
sns.distplot(
    res["corr"],
    hist_kws=dict(alpha=0.4, zorder=1, linewidth=0),
    bins=30,
    kde_kws=dict(cut=0, lw=1, zorder=1, alpha=0.8),
    color=CrispyPlot.PAL_DTRACE[2],
    ax=ax,
)
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.set_xlabel("Pearson's R")
ax.set_ylabel("Density")
ax.set_title(f"Protein ~ Transcript (mean R={res['corr'].mean():.2f})")
plt.savefig(
    f"{RPATH}/0.ProteinGexp_corr_histogram.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Protein attenuation
#

pal = dict(zip(*(["Low", "High", "NA"], CrispyPlot.PAL_DTRACE.values())))

_, ax = plt.subplots(1, 1, figsize=(1.0, 1.5), dpi=600)
sns.boxplot(
    x="attenuation",
    y="corr",
    notch=True,
    data=res,
    order=["High", "Low", "NA"],
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=pal,
    showcaps=False,
    saturation=1,
    ax=ax,
)
ax.set_xlabel("Attenuation")
ax.set_ylabel("Transcript ~ Protein")
ax.set_title("Protein expression attenuated\nin tumour patient samples")
ax.axhline(0, ls="-", lw=0.1, c="black", zorder=0)
ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
plt.savefig(f"{RPATH}/0.ProteinGexp_corr_attenuation_boxplot.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


# Representative examples
#

for gene in ["VIM", "EGFR", "IDH2", "NRAS", "SMARCB1", "ERBB2", "STAG1", "STAG2", "TP53", "RAC1", "MET"]:
    plot_df = pd.concat(
        [
            prot.loc[gene, samples].rename("protein"),
            gexp.loc[gene, samples].rename("transcript"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_regression("protein", "transcript", plot_df)
    grid.set_axis_labels(f"Protein", f"Transcript")
    grid.ax_marg_x.set_title(gene)
    plt.savefig(f"{RPATH}/0.Protein_Transcript_scatter_{gene}.pdf", bbox_inches="tight")
    plt.close("all")


# Hexbin correlation with CRISPR
#

plot_df = pd.concat(
    [
        crispr.loc[genes, samples].unstack().rename("CRISPR"),
        gexp.loc[genes, samples].unstack().rename("Transcriptomics"),
        prot.loc[genes, samples].unstack().rename("Proteomics"),
    ],
    axis=1,
    sort=False,
).dropna().query("Proteomics < 10")


for x, y in [("Transcriptomics", "CRISPR"), ("Proteomics", "CRISPR")]:
    _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

    ax.hexbin(
        plot_df[x],
        plot_df[y],
        cmap="Spectral_r",
        gridsize=100,
        mincnt=1,
        bins="log",
        lw=0,
    )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

    ax.set_xlabel("Protein (intensities)" if x == "Proteomics" else "Transcript (voom)")
    ax.set_ylabel(f"{y} (scaled log2)")

    plt.savefig(f"{RPATH}/0.ProteinGexp_hexbin_{x}_{y}.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")
