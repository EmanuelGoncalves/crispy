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
from crispy.CRISPRData import CRISPRDataSet
from C50K import (
    clib_palette,
    clib_order,
    define_sgrnas_sets,
    estimate_ks,
    sgrnas_scores_scatter,
)


rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

# -
# CRISPR Project Score - Kosuke Yusa v1.1

ky_v11_data = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_data.counts.remove_low_counts(ky_v11_data.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)

# Sets of sgRNAs
sgrna_sets = define_sgrnas_sets(ky_v11_data.lib, ky_v11_fc, add_controls=True)

# Kolmogorov-Smirnov statistic guide comparison
ky_v11_ks = estimate_ks(ky_v11_fc, sgrna_sets["nontargeting"]["fc"])


# CRISPR Library sizes

clib_size = pd.read_excel(f"{rpath}/Human CRISPR-Cas9 dropout libraries.xlsx")

plt.figure(figsize=(2.5, 2), dpi=600)
sns.scatterplot(
    "Date",
    "Number of guides",
    "Library ID",
    data=clib_size,
    size="sgRNAs per Gene",
    palette=clib_palette,
    hue_order=clib_order,
    sizes=(10, 50),
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
plt.xlim(datetime.date(2013, 11, 1), datetime.date(2020, 3, 1))
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 4})
plt.title("CRISPR-Cas9 libraries")
plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/crispr_libraries_coverage.pdf", bbox_inches="tight")
plt.close("all")


# Guide sets distributions

plt.figure(figsize=(2.5, 2), dpi=600)
for c in sgrna_sets:
    sns.distplot(
        sgrna_sets[c]["fc"],
        hist=False,
        label=c,
        kde_kws={"cut": 0, "shade": True},
        color=sgrna_sets[c]["color"],
    )
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("sgRNAs fold-change")
plt.legend(frameon=False)
plt.title("Project Score - KY v1.1")
plt.savefig(f"{rpath}/ky_v11_guides_distributions.pdf", bbox_inches="tight")
plt.close("all")


# K-S metric

sgrnas_scores_scatter(ky_v11_ks)

for gset in sgrna_sets:
    plt.axhline(
        sgrna_sets[gset]["fc"].median(),
        ls="-",
        lw=1,
        c=sgrna_sets[gset]["color"],
        zorder=0,
    )

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
plt.xlabel("sgRNA Kolmogorov-Smirnov (non-targeting)")
plt.ylabel("sgRNA median Fold-change")
plt.title("Project Score - KY v1.1")
plt.savefig(f"{rpath}/ky_v11_ks_scatter_median.png", bbox_inches="tight")
plt.close("all")


#

f, axs = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(4.5, 1.5), dpi=600)

for i, gset in enumerate(sgrna_sets):
    ax = axs[i]
    sgrnas_scores_scatter(
        ky_v11_ks,
        z=sgrna_sets[gset]["sgrnas"],
        z_color=sgrna_sets[gset]["color"],
        ax=ax,
    )
    for gset in sgrna_sets:
        ax.axhline(
            sgrna_sets[gset]["fc"].median(),
            ls="-",
            lw=1,
            c=sgrna_sets[gset]["color"],
            zorder=0,
        )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_title(gset)
    ax.set_xlabel("sgRNA KS\n(non-targeting)")
    ax.set_ylabel("sgRNAs fold-change\n(median)" if i == 0 else None)

plt.suptitle("Project Score - KY v1.1", y=1.07)
plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/ky_v11_ks_scatter_median_guides.png", bbox_inches="tight")
plt.close("all")
