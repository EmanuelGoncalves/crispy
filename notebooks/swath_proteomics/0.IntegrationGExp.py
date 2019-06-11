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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR


dpath = pkg_resources.resource_filename("crispy", "data")
rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")
logger = logging.getLogger("Crispy")


# # GDSC baseline cancer cell lines SWATH-Proteomics analysis

prot_obj = Proteomics()
gexp_obj = GeneExpression()
crispr_obj = CRISPR()


# Proteomics measurements

prot = prot_obj.get_data("imputed", True, True)
prot_ = prot_obj.get_data("noimputed", True, True)

# Transcriptomics measurements

trans = gexp_obj.get_data("voom")


# CRISPR-Cas0 measurements

crispr = crispr_obj.get_data()


# Overlap

genes = list(set(prot.index).intersection(trans.index))
logger.info(
    f"Genes: "
    f"Prot={len(set(prot.index))}; "
    f"Trans={len(set(trans.index))}; "
    f"CRISPR={len(set(crispr.index))}; "
    f"Overlap={len(genes)}"
)

samples = list(set(prot).intersection(trans))
logger.info(
    f"Genes: "
    f"Prot={len(set(prot))}; "
    f"Trans={len(set(trans))}; "
    f"CRISPR={len(set(crispr))}; "
    f"Overlap={len(samples)}"
)


#

for gene in ["VIM", "EGFR", "SFN", "SPINT1"]:
    plot_df = pd.concat(
        [
            trans.loc[gene, samples].rename("transcript"),
            prot.loc[gene, samples].rename("imputation"),
            prot_.loc[gene, samples].rename("no_imputation"),
        ],
        axis=1,
    )

    plot_df = pd.melt(
        plot_df, id_vars="transcript", value_vars=["imputation", "no_imputation"]
    ).dropna()

    g = sns.FacetGrid(plot_df, col="variable", sharex=True, sharey=False, despine=False)
    g.map(
        sns.regplot,
        "transcript",
        "value",
        truncate=True,
        line_kws=CrispyPlot.LINE_KWS,
        scatter_kws=CrispyPlot.SCATTER_KWS,
    )

    g.fig.suptitle(gene, y=1.05)
    g.fig.set_size_inches(4, 2)
    plt.savefig(f"{rpath}/gexp_corr_{gene}.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")


# Systematic correlation between gene-expression and protein abundance

gene_corr = []
for dtype, dataset in [("imputation", prot), ("no_imputation", prot_)]:
    logger.info(dtype)

    for g in genes:
        df = pd.concat(
            [
                trans.loc[g, samples].rename("transcript"),
                dataset.loc[g, samples].rename("protein"),
            ],
            axis=1,
        ).dropna()

        if df.shape[0] < 10:
            continue

        c, p = spearmanr(df["transcript"], df["protein"])

        res = dict(gene=g, corr=c, pval=p, len=df.shape[0], dtype=dtype)

        gene_corr.append(res)

gene_corr = pd.DataFrame(gene_corr).sort_values("pval")
gene_corr.to_csv(f"{dpath}/sl/prot_gexp_corr.csv.gz", index=False)
logger.info(gene_corr.groupby("dtype")["corr"].median())


#

g = sns.FacetGrid(gene_corr, col="dtype", sharex=True, sharey=False, despine=False)
g.map(sns.distplot, "corr", color=CrispyPlot.PAL_DBGD[0], kde_kws=dict(cut=0))
g.fig.set_size_inches(4, 2)
plt.savefig(f"{rpath}/gexp_corr_histograms.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


#

gene_corr_factors = pd.pivot_table(
    gene_corr, index="gene", columns="dtype", values="corr"
).dropna()

gene_corr_factors = pd.concat(
    [
        gene_corr_factors,
        prot_.mean(1).rename("mean"),
        prot_.count(1).rename("n_measurements"),
        prot_obj.count_peptides().rename("n_peptides"),
    ],
    axis=1,
    sort=False,
).dropna()


#

sns.regplot(
    "imputation",
    "no_imputation",
    gene_corr_factors,
    truncate=True,
    line_kws=CrispyPlot.LINE_KWS,
    scatter_kws=CrispyPlot.SCATTER_KWS,
)

plt.xlabel("imputation")
plt.ylabel("no_imputation")

plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/gexp_corr_datasets_regplot.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


#

plot_df = pd.melt(
    gene_corr_factors.reset_index(),
    id_vars=["mean", "n_measurements", "n_peptides", "index"],
    value_vars=["imputation", "no_imputation"],
    var_name="imputation",
    value_name="corr",
)

plot_df = pd.melt(
    plot_df,
    id_vars=["index", "imputation", "corr"],
    value_vars=["mean", "n_measurements", "n_peptides"],
    var_name="factor",
    value_name="value",
)

g = sns.FacetGrid(
    plot_df, col="imputation", row="factor", sharex=False, sharey=False, despine=False
)

g.map(
    sns.regplot,
    "corr",
    "value",
    color=CrispyPlot.PAL_DBGD[0],
    truncate=True,
    line_kws=CrispyPlot.LINE_KWS,
    scatter_kws=CrispyPlot.SCATTER_KWS,
)

g.fig.set_size_inches(6, 8)
plt.savefig(f"{rpath}/gexp_corr_factors.pdf", bbox_inches="tight", transparent=True)
plt.close("all")

# Copyright (C) 2019 Emanuel Goncalves
