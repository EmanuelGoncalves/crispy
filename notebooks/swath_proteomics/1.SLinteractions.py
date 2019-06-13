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
from scipy.stats import gaussian_kde
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import normaltest, shapiro
from statsmodels.stats.multitest import multipletests
from notebooks.swath_proteomics.SLethal import SLethal


logger = logging.getLogger("Crispy")
dpath = pkg_resources.resource_filename("crispy", "data")
rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


#

cgenes = pd.read_csv(f"{dpath}/cancer_genes_latest.csv.gz")


# Import data-sets

slethal = SLethal()


#

genes = set.intersection(set(slethal.gexp.index), set(slethal.prot.index))
logger.info(f"Genes={len(genes)}")


#

k_gexp = slethal.kinship(slethal.gexp_obj.get_data()[slethal.samples].T)
gi_gexp = slethal.genetic_interactions(slethal.gexp.loc[genes], k=k_gexp)

k_prot = slethal.kinship(slethal.prot_obj.get_data(dtype="imputed")[slethal.samples].T)
gi_prot = slethal.genetic_interactions(slethal.prot.loc[genes], k=k_prot)


#

gis = pd.concat(
    [
        gi_gexp.set_index(["phenotype", "gene"]).add_suffix("_gexp"),
        gi_prot.set_index(["phenotype", "gene"]).add_suffix("_prot"),
    ],
    axis=1,
    sort=False,
)
gis.to_csv(f"{dpath}/sl/gis_gexp_prot.csv.gz", compression="gzip")


#

plt.figure(figsize=(2.5, 2), dpi=600)

g = plt.hexbin(
    gis["beta_gexp"],
    gis["beta_prot"],
    cmap="Spectral_r",
    gridsize=30,
    mincnt=1,
    bins="log",
    linewidths=0,
)

plt.colorbar(g, spacing="uniform", extend="max")

plt.xlabel(f"CRISPR-Cas9 ~ RNA-Seq")
plt.ylabel(f"SWATH-MS ~ RNA-Seq")

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(f"{rpath}/gi_betas_hexbin.png", bbox_inches="tight")
plt.close("all")


#

plot_df = gis.query("(adjpval_gexp < .1) | (adjpval_prot < .1)")

xy = np.vstack([plot_df["beta_gexp"], plot_df["beta_prot"]])
z = gaussian_kde(xy)(xy)

grid = sns.JointGrid("beta_gexp", "beta_prot", data=plot_df, space=0)

grid.ax_joint.scatter(
    x=plot_df["beta_gexp"],
    y=plot_df["beta_prot"],
    edgecolor="w",
    lw=0,
    s=3,
    c=z,
    cmap="Spectral_r",
    alpha=1.,
)

grid.plot_marginals(
    sns.distplot, kde=False, hist_kws=dict(linewidth=0), color=CrispyPlot.PAL_DBGD[0]
)

grid.ax_joint.axhline(0, ls="-", lw=0.1, c=CrispyPlot.PAL_DBGD[0], zorder=0)
grid.ax_joint.axvline(0, ls="-", lw=0.1, c=CrispyPlot.PAL_DBGD[0], zorder=0)

(x0, x1), (y0, y1) = grid.ax_joint.get_xlim(), grid.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
grid.ax_joint.plot(lims, lims, ls="--", lw=0.3, zorder=0, c=CrispyPlot.PAL_DBGD[0])

grid.set_axis_labels(
    f"LMM effect size (beta)\nCRISPR-Cas9 ~ RNA-Seq",
    f"LMM effect size (beta)\nCRISPR-Cas9 ~ SWATH-MS",
)

plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig(f"{rpath}/gi_betas_signif_corr.pdf", bbox_inches="tight")
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
        df["crispr"],
        df[f],
        cmap="Spectral_r",
        gridsize=30,
        mincnt=1,
        bins="log",
        linewidths=0,
    )

    plt.colorbar(g, spacing="uniform", extend="max")

    plt.xlabel(f"CRISPR-Cas9")
    plt.ylabel("RNA-Seq" if f == "gexp" else "SWATH-MS")

    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    plt.savefig(f"{rpath}/essentiality_correlation_{f}.png", bbox_inches="tight")
    plt.close("all")


#

f, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)

g = ax.hexbin(
    df["gexp"],
    df["prot"],
    cmap="Spectral_r",
    gridsize=30,
    mincnt=1,
    bins="log",
    linewidths=0,
)

plt.colorbar(g, spacing="uniform", extend="max")

plt.xlabel(f"RNA-Seq")
plt.ylabel("SWATH-MS")

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

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
