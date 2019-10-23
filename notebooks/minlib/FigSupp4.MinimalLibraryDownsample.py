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
import matplotlib.pyplot as plt
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib.Utils import project_score_sample_map, downsample_sgrnas, density_interpolate


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# KY v1.1 library
#

ky_lib = Library.load_library("MasterLib_v1.csv.gz").query("Library == 'KosukeYusa'")
ky_lib = ky_lib[ky_lib.index.isin(ky_counts.index)]


# Minimal library (top 2)
#

NGUIDES, REMOVE_DISCORDANT = 2, True
ml_lib_name = f"MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz"
ml_lib = Library.load_library(ml_lib_name).query("Library == 'KosukeYusa'")
ml_lib = ml_lib[ml_lib.index.isin(ky_counts.index)]
ml_lib = ml_lib.loc[[i for i in ml_lib.index if not i.startswith("CTRL0")]]


# Genes overlap
#

genes = set(ky_lib["Approved_Symbol"]).intersection(ml_lib["Approved_Symbol"])
LOG.info(f"Genes={len(genes)}")


# Downsample analysis
#

ky_scores = downsample_sgrnas(ky_counts, ky_lib, ky_smap, [2], gene_col="Approved_Symbol")
ml_scores = downsample_sgrnas(ky_counts, ml_lib, ky_smap, [2], gene_col="Approved_Symbol")


# Plot density scatter
#

plot_df = pd.concat([
    ky_scores.groupby(["sample", "n_guides"])["aroc"].median().rename("Random"),
    ml_scores.groupby(["sample", "n_guides"])["aroc"].median().rename("MinimalLib"),
], axis=1, sort=False).reset_index()
plot_df["density"] = density_interpolate(plot_df["Random"], plot_df["MinimalLib"])
plot_df = plot_df.sort_values("density")

x_min, x_max = plot_df.iloc[:, [2, 3]].min().min(), plot_df.iloc[:, [2, 3]].max().max()

plt.scatter(
    plot_df["Random"],
    plot_df["MinimalLib"],
    c=plot_df["density"],
    marker="o",
    edgecolor="",
    cmap="Spectral_r",
    s=3,
    alpha=0.7,
)

lims = [x_max, x_min]
plt.plot(lims, lims, "k-", lw=0.3, zorder=0)

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Random - 2 sgRNAs")
plt.ylabel("Minimal Library - 2 sgRNAs")
plt.title("AROC essential genes")

plt.gcf().set_size_inches(2, 2)
plt.savefig(f"{RPATH}/MinimalLib_downsample_scatter.pdf", bbox_inches="tight", transparent=True)
plt.close("all")
