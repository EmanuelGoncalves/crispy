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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet


if __name__ == "__main__":
    data = CRISPRDataSet("Yusa_v1.1")

    # -
    fc = (
        data.counts.remove_low_counts(data.plasmids)
        .norm_rpm()
        .foldchange(data.plasmids)
    )

    # -
    jacks = pd.read_csv("/Users/eg14/Data/gdsc/crispr/grna_efficacy_v1.1.tab", index_col=0, sep="\t")
    jacks_wh = pd.read_csv("/Users/eg14/Downloads/whitehead_grna_JACKS_results.txt", index_col=0, sep="\t")

    # -
    sgrnas_intersection = pd.concat([
        jacks["Efficacy"].rename("Yusa_1.1"),
        jacks_wh["X1"].rename("Sabatini_Lander")
    ], sort=False, axis=1).dropna()

    (sgrnas_intersection["Yusa_1.1"] - sgrnas_intersection["Sabatini_Lander"]).hist(bins=50); plt.show()

    # -
    # gene = "PIK3CA"
    sgrnas_correlation = []
    for gene in set(data.lib["Gene"]):
        print(gene)

        sgrnas = list(set(data.lib[data.lib["Gene"] == gene].index).intersection(fc.index))

        if len(sgrnas) > 1:
            sgrnas_fc = fc.loc[sgrnas].T

            mean_corr = sgrnas_fc.corr()
            mean_corr_keep = (
                np.triu(np.ones(mean_corr.shape), 1).astype("bool").reshape(mean_corr.size)
            )
            mean_corr = mean_corr.stack()[mean_corr_keep].mean()

            sgrnas_corr = sgrnas_fc.corrwith(sgrnas_fc.mean(1))

            sgrnas_res = (
                sgrnas_corr.rename("corr")
                .to_frame()
                .assign(gene=gene)
                .assign(mean_corr=mean_corr)
                .assign(len=sgrnas_fc.shape[1])
            )

            sgrnas_correlation.append(sgrnas_res)
    sgrnas_correlation = pd.concat(sgrnas_correlation)

    # -
    sgrnas_correlation = pd.concat([sgrnas_correlation, jacks["Efficacy"].rename("jacks")], axis=1, sort=False).dropna()

    # -
    plt.figure(figsize=(1.5, 1.5), dpi=300)
    plt.scatter(sgrnas_correlation["corr"], sgrnas_correlation["mean_corr"])
    plt.show()

    # -
    pearsonr(sgrnas_correlation["corr"], sgrnas_correlation["jacks"])

    plt.figure(figsize=(1.5, 1.5), dpi=300)
    plt.scatter(sgrnas_correlation["corr"], sgrnas_correlation["jacks"], s=3, edgecolors="white", lw=.1, alpha=.5, c=CrispyPlot.PAL_DBGD[0])
    plt.show()


# Copyright (C) 2019 Emanuel Goncalves
