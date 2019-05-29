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

import datetime
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from crispy.QCPlot import QCplot
from crispy.Utils import Utils
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error
from crispy.CopyNumberCorrection import Crispy
from scipy.stats import spearmanr, mannwhitneyu
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from crispy.DataImporter import Mobem, CopyNumber, CopyNumberSegmentation
from C50K import (
    clib_palette,
    clib_order,
    define_sgrnas_sets,
    estimate_ks,
    sgrnas_scores_scatter,
    guides_aroc_benchmark,
    ky_v11_calculate_gene_fc,
    lm_associations,
    project_score_sample_map,
    assemble_crisprcleanr,
    assemble_crispy,
)


# -
dpath = pkg_resources.resource_filename("crispy", "data/")
rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")


# Gene sets fold-change distributions

genesets = {
    "essential": Utils.get_essential_genes(return_series=False),
    "nonessential": Utils.get_non_essential_genes(return_series=False),
}

genesets = {
    m: {g: metrics_fc[m].reindex(genesets[g]).median(1).dropna() for g in genesets}
    for m in metrics_fc
}




# Gene sets distributions

f, axs = plt.subplots(
    1,
    len(metrics_fc),
    sharex="all",
    sharey="all",
    figsize=(len(metrics_fc) * 1.5, 1.5),
    dpi=600,
)

for i, m in enumerate(genesets):
    ax = axs[i]

    s, p = mannwhitneyu(genesets[m]["essential"], genesets[m]["nonessential"])

    for n in genesets[m]:
        sns.distplot(
            genesets[m][n],
            hist=False,
            label=n,
            kde_kws={"cut": 0, "shade": True},
            color=sgrna_sets[n]["color"],
            ax=ax,
        )
        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
        ax.set_title(f"{m}")
        ax.set_xlabel("sgRNAs fold-change")
        ax.text(
            0.05,
            0.75,
            f"Mann-Whitney p-val={p:.2g}",
            fontsize=4,
            transform=ax.transAxes,
            ha="left",
        )
        ax.legend(frameon=False, prop={"size": 4}, loc=2)

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_metrics_gene_distributions.png", bbox_inches="tight")
plt.close("all")

