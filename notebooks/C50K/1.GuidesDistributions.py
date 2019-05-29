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
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from C50K import rpath, define_sgrnas_sets
from crispy.CRISPRData import CRISPRDataSet


# Project Score - Kosuke_Yusa v1.1

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_count.counts.remove_low_counts(ky_v11_count.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_count.plasmids)
)


# Broad DepMap19Q2 - Avana

avana_data = CRISPRDataSet("Avana")
avana_fc = (
    avana_data.counts.remove_low_counts(avana_data.plasmids)
    .norm_rpm()
    .foldchange(avana_data.plasmids)
)


# Plot sgRNA sets of Yusa_v1.1

sgrna_sets = define_sgrnas_sets(ky_v11_count.lib, ky_v11_fc, add_controls=True)

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
plt.show()


# Plot sgRNA sets of Avana

avana_sgrna_set = define_sgrnas_sets(
    avana_data.lib, avana_fc, add_controls=True, dataset_name="Avana"
)

plt.figure(figsize=(2.5, 2), dpi=600)
for c in avana_sgrna_set:
    sns.distplot(
        avana_sgrna_set[c]["fc"],
        hist=False,
        label=c,
        kde_kws={"cut": 0, "shade": True},
        color=avana_sgrna_set[c]["color"],
    )
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("sgRNAs fold-change")
plt.legend(frameon=False)
plt.title("DepMap19Q2 - Avana")
plt.savefig(f"{rpath}/avana_guides_distributions.pdf", bbox_inches="tight")
plt.close("all")
