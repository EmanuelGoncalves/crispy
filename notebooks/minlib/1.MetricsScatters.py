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

import pandas as pd
import matplotlib.pyplot as plt
from crispy.CRISPRData import CRISPRDataSet
from minlib import rpath, sgrnas_scores_scatter, define_sgrnas_sets


# Guides efficacy metric

ky_v11_metrics = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0)


# Project Score - Kosuke_Yusa v1.1

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_count.counts.remove_low_counts(ky_v11_count.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_count.plasmids)
)

sgrna_sets = define_sgrnas_sets(ky_v11_count.lib, ky_v11_fc, add_controls=True)


# KS-control metric versus median fold-change scatter

sgrnas_scores_scatter(ky_v11_metrics)

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
plt.show()


# K-S metric guides distribution

f, axs = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(4.5, 1.5), dpi=600)

for i, gset in enumerate(sgrna_sets):
    ax = axs[i]
    sgrnas_scores_scatter(
        ky_v11_metrics,
        z=sgrna_sets[gset]["sgrnas"],
        z_color=sgrna_sets[gset]["color"],
        ax=ax,
    )
    for g in sgrna_sets:
        ax.axhline(
            sgrna_sets[g]["fc"].median(),
            ls="-",
            lw=1,
            c=sgrna_sets[g]["color"],
            zorder=0,
        )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_title(gset)
    ax.set_xlabel("sgRNA KS\n(non-targeting)")
    ax.set_ylabel("sgRNAs fold-change\n(median)" if i == 0 else None)

plt.suptitle("Project Score - KY v1.1", y=1.07)
plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/ky_v11_ks_scatter_median_guides.png", bbox_inches="tight")
plt.show()


# Metrics correlation

f, axs = plt.subplots(1, 4, sharex="none", sharey="all", figsize=(6.0, 1.5), dpi=600)

for i, (x, y, n) in enumerate(
    [
        ("ks_control", "median_fc", "K-S statistic"),
        ("jacks", "median_fc", "JACKS scores"),
        ("doenchroot", "median_fc", "Doench-Root score"),
        ("forecast", "median_fc", "FORECAST\nIn Frame Percentage"),
    ]
):
    ax = axs[i]

    sgrnas_scores_scatter(ky_v11_metrics, x, y, add_cbar=False, ax=ax)

    for g in sgrna_sets:
        ax.axhline(
            sgrna_sets[g]["fc"].median(),
            ls="-",
            lw=1,
            c=sgrna_sets[g]["color"],
            zorder=0,
        )

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_xlabel(n)
    ax.set_ylabel(None if i != 0 else "sgRNAs fold-change\n(median)")

plt.suptitle("Project Score - KY v1.1", y=1.07)
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_ks_scatter_metrics.png", bbox_inches="tight")
plt.show()
