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


rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")

# # GDSC baseline cancer cell lines SWATH-Proteomics analysis

data = Proteomics()

dtypes = ["peptide", "protein", "imputed"]


# ## Initial QC

# Number of replicates of cell line per machine

machine = pd.pivot_table(data.manifest, index="CellLine", columns="Machine", aggfunc=len, fill_value=0)["Date"]
machine["Total"] = machine.sum(1)
machine.sort_values("Total").head(10)


# Plot the distributions of the peptides intensites

for dtype in dtypes:
    f, axs = plt.subplots(2, 1, dpi=300, figsize=(12, 3), sharex="none", sharey="all")

    for i, (m, df) in enumerate(data.manifest.groupby("Machine")):
        plot_df = data.get_data(dtype)[df.index].dropna(how="all")
        plot_df = np.log10(plot_df)

        sns.boxplot(
            data=plot_df,
            boxprops=CrispyPlot.BOXPROPS,
            whiskerprops=CrispyPlot.WHISKERPROPS,
            medianprops=CrispyPlot.MEDIANPROPS,
            flierprops=CrispyPlot.FLIERPROPS,
            color=CrispyPlot.PAL_DBGD[0],
            ax=axs[i]
        )

        axs[i].axes.xaxis.set_visible(False)

    plt.savefig(
        f"{rpath}/{dtype}_counts_boxplots.png", bbox_inches="tight", transparent=True
    )
    plt.close("all")


# Sample correlation across cell lines, cancer type and machine

corr = {}
for dtype in dtypes:
    samples_corr = np.log10(data.get_data(dtype)).corr()

    keep = np.triu(np.ones(samples_corr.shape), 1).astype("bool").reshape(samples_corr.size)
    samples_corr = samples_corr.stack()[keep].reset_index()

    samples_corr = samples_corr.rename(
        columns={"level_0": "sample1", "level_1": "sample2", 0: "pearson"}
    )

    headers = ["CellLine", "Tissue", "Cancer", "Machine"]
    samples_corr = pd.concat([
        samples_corr,
        data.manifest.loc[samples_corr["sample1"], headers].add_suffix("_1").reset_index(drop=True),
        data.manifest.loc[samples_corr["sample2"], headers].add_suffix("_2").reset_index(drop=True),
    ], axis=1, sort=False)

    samples_corr["same_cell_line"] = (samples_corr["CellLine_1"] == samples_corr["CellLine_2"]).astype(int)
    samples_corr["same_cancer_type"] = (samples_corr["Cancer_1"] == samples_corr["Cancer_2"]).astype(int)
    samples_corr["same_machine"] = (samples_corr["Machine_1"] == samples_corr["Machine_2"]).astype(int)

    corr[dtype] = samples_corr


# Sample correlation boxplots

for dtype in dtypes:
    comparisons = ["same_cell_line", "same_cancer_type", "same_machine"]

    f, axs = plt.subplots(1, len(comparisons), dpi=300, figsize=(2, 1.5), sharex="none", sharey="all")
    for i, n in enumerate(comparisons):
        label = n[5:]

        sns.boxplot(
            x=n,
            y="pearson",
            data=corr[dtype],
            boxprops=CrispyPlot.BOXPROPS,
            whiskerprops=CrispyPlot.WHISKERPROPS,
            medianprops=CrispyPlot.MEDIANPROPS,
            flierprops=CrispyPlot.FLIERPROPS,
            palette=CrispyPlot.PAL_DBGD,
            showcaps=False,
            ax=axs[i]
        )

        axs[i].set_title(label)

        axs[i].set_xlabel("Replicate")

        if i != 0:
            axs[i].set_ylabel("")

    plt.suptitle(dtype, y=1.05)
    plt.subplots_adjust(wspace=0.075, hspace=0.075)
    plt.savefig(
        f"{rpath}/{dtype}_corr_boxplot.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")


# Copyright (C) 2019 Emanuel Goncalves
