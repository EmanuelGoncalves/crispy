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

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from GIPlot import GIPlot
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr, skew
from minlib.Utils import density_interpolate
from sklearn.metrics import mean_squared_error
from minlib.Utils import project_score_sample_map
from crispy.CRISPRData import CRISPRDataSet, Library
from crispy.LibRepresentationReport import LibraryRepresentaion


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# MinLibCas9 library information
#

mlib = Library.load_library("MinLibCas9.csv.gz", set_index=False)
mlib.index = [f"{i}" for i in mlib["WGE_ID"]]
mlib["sgRNA"] = [s if len(s) == 19 else s[1:-3] for s in mlib["WGE_Sequence"]]


# Assemble raw counts matrix
#

SPATH = pkg_resources.resource_filename("notebooks", "minlib/minlibcas9_screens")

plasmid_counts = pd.read_csv(f"{SPATH}/Minimal_library_output_108.csv", index_col=0).rename(columns=dict(counts="MinLibCas9"))


#
#

lib_report = LibraryRepresentaion(plasmid_counts[["MinLibCas9"]])
pal = dict(MHG_library_v1=QCplot.PAL_DBGD[0], MinLibCas9=QCplot.PAL_DBGD[1])

# Lorenz curves#
lib_report.lorenz_curve(palette=pal)
plt.gcf().set_size_inches(2., 2.)
plt.savefig(f"{RPATH}/librepresentation_lorenz_curve.pdf", bbox_inches="tight", dpi=600)
plt.close("all")

# Lorenz curves#
plot_df = plasmid_counts["MinLibCas9"].sort_values().reset_index()

skew_ratio = plot_df["MinLibCas9"].quantile([.9, .1])
skew_ratio = skew_ratio[.9] / skew_ratio[.1]

fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5), dpi=600)

ax.plot(
    plot_df.index,
    plot_df["MinLibCas9"],
    color=pal["MinLibCas9"],
    # edgecolor="w",
    lw=1,
    # s=6,
    alpha=.8,
    zorder=3,
)

ax.set_xlabel("Ranked sgRNAs")
ax.set_ylabel("Number of reads")

ax.set_xticks([0, plot_df.shape[0] / 2, plot_df.shape[0]])

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

annot_text = f"Skew ratio = {skew_ratio:.2f}"
ax.text(
    0.95,
    0.05,
    annot_text,
    fontsize=6,
    transform=ax.transAxes,
    ha="right",
)

plt.savefig(f"{RPATH}/librepresentation_scatter.pdf", bbox_inches="tight", dpi=600)
plt.close("all")
