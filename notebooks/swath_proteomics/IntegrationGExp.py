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
from scipy.stats import spearmanr
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics, GeneExpression, CRISPRComBat


rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")
logger = logging.getLogger("Crispy")


# # GDSC baseline cancer cell lines SWATH-Proteomics analysis

prot_obj = Proteomics()
gexp_obj = GeneExpression()
crispr_obj = CRISPRComBat()


# Proteomics measurements

prot = prot_obj.get_data("imputed", True, True)
prot_ = prot_obj.get_data("protein", True, True)

# Transcriptomics measurements

trans = gexp_obj.get_data("voom")


# CRISPR-Cas0 measurements

crispr = crispr_obj.get_data()


# Overlap

genes = list(set(prot.index).intersection(trans.index))
logger.log(
    logging.INFO,
    f"Genes: "
    f"Prot={len(set(prot.index))}; "
    f"Trans={len(set(trans.index))}; "
    f"CRISPR={len(set(crispr.index))}; "
    f"Overlap={len(genes)}",
)

samples = list(set(prot).intersection(trans))
logger.log(
    logging.INFO,
    f"Genes: "
    f"Prot={len(set(prot))}; "
    f"Trans={len(set(trans))}; "
    f"CRISPR={len(set(crispr))}; "
    f"Overlap={len(samples)}",
)


#

for gene in ["VIM", "EGFR"]:
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


#

gene_corr = []
for g in genes:
    df = pd.concat(
        [
            trans.loc[g, samples].rename("transcript"),
            prot.loc[g, samples].rename("protein"),
        ],
        axis=1,
    ).dropna()

    if df.shape[0] < 10:
        continue

    c, p = spearmanr(df["transcript"], df["protein"])

    res = dict(gene=g, corr=c, pval=p, len=df.shape[0])

    gene_corr.append(res)

gene_corr = pd.DataFrame(gene_corr).sort_values("pval")

print(gene_corr["corr"].median())

plt.figure(figsize=(1.5, 1.5), dpi=300)
gene_corr["corr"].hist(bins=30, linewidth=0)
plt.show()
