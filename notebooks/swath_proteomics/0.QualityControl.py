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

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics
from sklearn.preprocessing import quantile_transform


rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")

# # GDSC baseline cancer cell lines SWATH-Proteomics analysis

proteomics = Proteomics()

protein_reps = proteomics.get_data(average_replicates=False)
protein = proteomics.get_data(average_replicates=False)


# ## Initial QC

# Number of replicates of cell line per machine

machine = pd.pivot_table(
    proteomics.manifest, index="SIDM", columns="Instrument", aggfunc=len, fill_value=0
)["Date"]
machine["Total"] = machine.sum(1)
machine.sort_values("Total").head(10)


# Plot the distributions of the peptides intensites

n_rows = len(set(proteomics.manifest["Instrument"]))
f, axs = plt.subplots(
    n_rows, 1, dpi=300, figsize=(12, 2 * n_rows), sharex="none", sharey="all"
)

for i, (m, df) in enumerate(proteomics.manifest.groupby("Instrument")):
    plot_df = protein_reps.reindex(columns=df.index).dropna(how="all")

    sns.boxplot(
        data=plot_df,
        boxprops=CrispyPlot.BOXPROPS,
        whiskerprops=CrispyPlot.WHISKERPROPS,
        medianprops=CrispyPlot.MEDIANPROPS,
        flierprops=CrispyPlot.FLIERPROPS,
        color=CrispyPlot.PAL_DBGD[0],
        ax=axs[i],
    )

    axs[i].set_ylabel(f"{m}")
    axs[i].axes.xaxis.set_visible(False)

plt.savefig(
    f"{rpath}/qc_protein_counts_boxplots.png", bbox_inches="tight", transparent=True
)
plt.close("all")


# Sample correlation across cell lines, cancer type and machine

samples_corr = protein_reps.corr()
samples_corr.columns.name = "sample2"
samples_corr.index.name = "sample1"

keep = np.triu(np.ones(samples_corr.shape), 1).astype("bool").reshape(samples_corr.size)
samples_corr = samples_corr.stack()[keep].reset_index()

samples_corr = samples_corr.rename(columns={0: "pearson"})

headers = ["Cell_line", "Tissue_type", "Cancer_type", "Instrument"]
samples_corr = pd.concat(
    [
        samples_corr,
        proteomics.manifest.loc[samples_corr["sample1"], headers]
        .add_suffix("_1")
        .reset_index(drop=True),
        proteomics.manifest.loc[samples_corr["sample2"], headers]
        .add_suffix("_2")
        .reset_index(drop=True),
    ],
    axis=1,
    sort=False,
)

samples_corr["same_cell_line"] = (
    samples_corr["Cell_line_1"] == samples_corr["Cell_line_2"]
).astype(int)
samples_corr["same_cancer_type"] = (
    samples_corr["Tissue_type_1"] == samples_corr["Tissue_type_2"]
).astype(int)
samples_corr["same_machine"] = (
    samples_corr["Instrument_1"] == samples_corr["Instrument_2"]
).astype(int)

samples_corr.to_csv(f"{rpath}/protein_corr.csv", compression="gzip", index=False)


# Sample correlation boxplots

comparisons = ["same_cell_line", "same_cancer_type", "same_machine"]

f, axs = plt.subplots(
    1, len(comparisons), dpi=600, figsize=(2, 1.5), sharex="none", sharey="all"
)
for i, n in enumerate(comparisons):
    label = n[5:]

    sns.boxplot(
        x=n,
        y="pearson",
        data=samples_corr,
        boxprops=CrispyPlot.BOXPROPS,
        whiskerprops=CrispyPlot.WHISKERPROPS,
        medianprops=CrispyPlot.MEDIANPROPS,
        flierprops=CrispyPlot.FLIERPROPS,
        palette=CrispyPlot.PAL_DBGD,
        showcaps=False,
        ax=axs[i],
    )

    axs[i].set_title(label)
    axs[i].set_xlabel("Replicate")
    if i != 0:
        axs[i].set_ylabel("")

    axs[i].grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.subplots_adjust(wspace=0.075, hspace=0.075)
plt.savefig(
    f"{rpath}/qc_protein_corr_boxplot.png", bbox_inches="tight", transparent=True
)
plt.close("all")


# Histogram of number of measurements per protein

_, ax = plt.subplots(1, 1, figsize=(2.5, 1.5), dpi=600)

sns.distplot(
    protein.count(1),
    color=CrispyPlot.PAL_DBGD[0],
    kde=False,
    hist_kws=dict(lw=0, alpha=1),
    label=None,
    ax=ax,
)

ax.set_xlabel("Number of measurements")
ax.set_ylabel("Number of proteins")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{rpath}/qc_protein_measurements_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Histogram of number of measurements per cell line

_, ax = plt.subplots(1, 1, figsize=(2.5, 1.5), dpi=600)

sns.distplot(
    protein.count(0),
    color=CrispyPlot.PAL_DBGD[0],
    kde=False,
    hist_kws=dict(lw=0, alpha=1),
    label=None,
    ax=ax,
)

ax.set_xlabel("Number of measurements")
ax.set_ylabel("Number of cell lines")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{rpath}/qc_protein_cell_lines_measurements_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Copyright (C) 2019 Emanuel Goncalves
