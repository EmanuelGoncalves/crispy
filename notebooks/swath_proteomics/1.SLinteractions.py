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
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import normaltest, shapiro
from statsmodels.stats.multitest import multipletests
from notebooks.swath_proteomics.SLethal import SLethal


logger = logging.getLogger("Crispy")
rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# # Import data-sets

slethal = SLethal()


#

genes = set.intersection(
    set(slethal.gexp.index), set(slethal.prot.index)
)
logger.info(f"Genes={len(genes)}")


#

gi_gexp = slethal.genetic_interactions(slethal.gexp.loc[genes])
gi_prot = slethal.genetic_interactions(slethal.prot.loc[genes])


#

gis = pd.concat([
    gi_gexp.set_index(["phenotype", "gene"]).add_suffix("_gexp"),
    gi_prot.set_index(["phenotype", "gene"]).add_suffix("_prot"),
], axis=1, sort=False)


#

plt.figure(figsize=(2.5, 2), dpi=600)

g = plt.hexbin(
    gis["beta_gexp"], gis["beta_prot"], cmap="Spectral_r", gridsize=30, mincnt=1, bins="log", linewidths=0
)

plt.colorbar(g, spacing="uniform", extend="max")

plt.xlabel(f"CRISPR-Cas9 ~ RNA-Seq")
plt.ylabel(f"SWATH-MS ~ RNA-Seq")

plt.grid(True, ls=":", lw=.1, alpha=1., zorder=0)

plt.savefig(f"{rpath}/gi_betas_hexbin.png", bbox_inches="tight")
plt.close("all")


#

df = pd.concat(
    [
        slethal.gexp.loc[genes, slethal.samples].unstack().rename("gexp"),
        slethal.prot.loc[genes, slethal.samples].unstack().rename("prot"),
        slethal.crispr.loc[genes, slethal.samples].unstack().rename("crispr"),
    ],
    axis=1,
    sort=False,
).dropna()


#

for f in ["gexp", "prot"]:
    plt.figure(figsize=(2.5, 2), dpi=600)

    g = plt.hexbin(
        df["crispr"], df[f], cmap="Spectral_r", gridsize=30, mincnt=1, bins="log", linewidths=0
    )

    plt.colorbar(g, spacing="uniform", extend="max")

    plt.xlabel(f"CRISPR-Cas9")
    plt.ylabel("RNA-Seq" if f == "gexp" else "SWATH-MS")

    plt.grid(True, ls=":", lw=.1, alpha=1., zorder=0)

    plt.savefig(f"{rpath}/essentiality_correlation_{f}.png", bbox_inches="tight")
    plt.close("all")


#

f, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)

g = ax.hexbin(
    df["gexp"], df["prot"], cmap="Spectral_r", gridsize=30, mincnt=1, bins="log", linewidths=0
)

plt.colorbar(g, spacing="uniform", extend="max")

plt.xlabel(f"RNA-Seq")
plt.ylabel("SWATH-MS")

plt.grid(True, ls=":", lw=.1, alpha=1., zorder=0)

plt.savefig(f"{rpath}/prot_gexp_correlation.png", bbox_inches="tight")
plt.close("all")


# #
#
# gi_gexp = slethal.genetic_interactions_gexp()
#
#
# #
#
# gene_phenotype, gene_feature = "PRDX1", "PRDX2"
# plot_df = pd.concat(
#     [
#         slethal.crispr.loc[gene_phenotype].rename("phenotype"),
#         slethal.gexp.loc[gene_feature].rename("gene"),
#     ],
#     axis=1,
#     sort=False,
# ).dropna()
#
# plt.figure(figsize=(1.5, 1.5), dpi=300)
# sns.regplot(
#     "phenotype",
#     "gene",
#     data=plot_df,
#     color=CrispyPlot.PAL_DBGD[0],
#     truncate=True,
#     line_kws=CrispyPlot.LINE_KWS,
#     scatter_kws=CrispyPlot.SCATTER_KWS,
# )
# plt.xlabel(f"{gene_phenotype} CRISPR-Cas9")
# plt.ylabel(f"{gene_feature} RNA-Seq")
# plt.show()

# Copyright (C) 2019 Emanuel Goncalves
