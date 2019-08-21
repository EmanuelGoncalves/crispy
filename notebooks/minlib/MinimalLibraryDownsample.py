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
from crispy.CRISPRData import CRISPRDataSet, Library, ReadCounts
from minlib.Utils import project_score_sample_map, downsample_sgrnas, density_interpolate


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Samples manifest
#

smap = project_score_sample_map()
plasmids = ["CRISPR_C6596666.sample"]


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky_counts = CRISPRDataSet("Yusa_v1.1").counts.remove_low_counts(plasmids)
ky_lib = Library.load_library("MasterLib_v1.csv.gz").query(
    "Library == 'KosukeYusa'"
)


# Minimal library (top 2)
#

ml_lib = Library.load_library("MinimalLib_top2.csv.gz").query(
    "Library == 'KosukeYusa'"
)
ml_counts = ReadCounts(
    pd.read_csv(
        f"{RPATH}/KosukeYusa_v1.1_sgrna_counts_MinimalLib_top2.csv.gz", index_col=0
    )
).remove_low_counts(plasmids)


# Genes overlap
#

genes = set(ky_lib["Approved_Symbol"]).intersection(ml_lib["Approved_Symbol"])
LOG.info(f"Genes={len(genes)}")

ky_lib = ky_lib[ky_lib["Approved_Symbol"].isin(genes)]
ml_lib = ml_lib[ml_lib["Approved_Symbol"].isin(genes)]


# Downsample analysis
#

ky_scores = downsample_sgrnas(ky_counts, ky_lib, smap, [2], gene_col="Approved_Symbol")
ml_scores = downsample_sgrnas(ml_counts, ml_lib, smap, [2], gene_col="Approved_Symbol")


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
plt.title("AROC essential genes\n1%FDR")

plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig(f"{RPATH}/MinimalLib_downsample_scatter.pdf", bbox_inches="tight")
plt.close("all")
