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
from swath_proteomics.GIPlot import GIPlot
from scipy.stats import normaltest, shapiro
from statsmodels.stats.multitest import multipletests
from notebooks.swath_proteomics.SLethal import SLethal


dpath = pkg_resources.resource_filename("crispy", "data")
rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


#

cgenes = pd.read_csv(f"{dpath}/cancer_genes_latest.csv.gz")


# Import data-sets

slethal = SLethal()


# CRISPR kinship matrix

k_crispr = slethal.kinship(slethal.crispr_obj.get_data()[slethal.samples].T)


# Overlap genes

genes = list(set(slethal.prot.index).intersection(slethal.gexp.index))


# Genetic interactions with protein abundance (gexp covariate)

gi_prot = slethal.genetic_interactions(
    y=slethal.prot.loc[genes], x=slethal.crispr, k=k_crispr, z=slethal.gexp
)
gi_prot.to_csv(f"{rpath}/gi_prot_crispr.csv.gz", compression="gzip", index=False)


#

g_protein, g_crispr = "SEC22B", "SOX10"

plot_df = pd.concat(
    [
        slethal.prot.loc[g_protein].rename("protein"),
        slethal.gexp.loc[g_protein].rename("transcript"),
        slethal.crispr.loc[g_crispr].rename("crispr"),
    ],
    axis=1,
    sort=False,
).dropna()

grid = GIPlot.gi_regression("protein", "crispr", plot_df)
grid.set_axis_labels(f"Protein {g_protein}", f"Essentiality {g_crispr}")
plt.savefig(
    f"{rpath}/protein_essentiality_scatter_{g_protein}_{g_crispr}.pdf",
    bbox_inches="tight",
)
plt.close("all")

grid = GIPlot.gi_regression("transcript", "crispr", plot_df)
grid.set_axis_labels(f"Transcript {g_protein}", f"Essentiality {g_crispr}")
plt.savefig(
    f"{rpath}/transcript_essentiality_scatter_{g_protein}_{g_crispr}.pdf",
    bbox_inches="tight",
)
plt.close("all")
