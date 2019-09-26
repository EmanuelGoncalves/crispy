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
from crispy import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib.Utils import project_score_sample_map, replicates_correlation


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

ml_lib = Library.load_library("MinimalLib_top2.csv.gz").query("Library == 'KosukeYusa'")
ml_lib = ml_lib[ml_lib.index.isin(ky_counts.index)]


# Comparisons
#

libraries = dict(
    All=dict(name="All", lib=ky_lib), Minimal=dict(name="Minimal", lib=ml_lib)
)


# sgRNA efficiency metrics
#

rep_corrs = []
for ltype in libraries:
    lib = libraries[ltype]["lib"]
    LOG.info(f"Library={ltype}; sgRNA={lib.shape[0]}")

    m_rep_corr = replicates_correlation(ky_counts, guides=list(lib.index)).assign(
        library=ltype
    )

    rep_corrs.append(m_rep_corr)

rep_corrs = pd.concat(rep_corrs)

rep_corrs.to_csv(
    f"{RPATH}/KosukeYusa_v1.1_benchmark_rep_correlation.csv.gz",
    compression="gzip",
    index=False,
)


# Plot replicates correlations boxplots
#

ltypes = rep_corrs.groupby("library")

palette = {True: "#d62728", False: "#E1E1E1"}

f, axs = plt.subplots(
    1, len(ltypes), sharex="all", sharey="row", figsize=(len(ltypes) * 0.75, 2.)
)

all_rep_corr = rep_corrs.query("(library == 'All') & (replicate == True)")[
    "corr"
].median()

for i, (dtype, df) in enumerate(ltypes):
    ax = axs[i]

    sns.boxplot(
        "replicate",
        "corr",
        data=df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        notch=True,
        ax=ax,
    )

    ax.axhline(all_rep_corr, ls="-", lw=.5, zorder=0, c="#484848")

    ax.set_title(dtype)
    ax.set_ylabel(f"Replicates correlation\n(Pearson R)" if i == 0 else None)

plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{RPATH}/Replicates_correlation_boxplots.pdf", bbox_inches="tight")
plt.close("all")
