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
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.CRISPRData import CRISPRDataSet
from sklearn.metrics import mean_squared_error
from C50K import dpath, rpath, define_sgrnas_sets, estimate_ks


# CRISPR Project Score - Kosuke Yusa v1.1

ky_v11_data = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_data.counts.remove_low_counts(ky_v11_data.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)
ky_v11_sgrna_set = define_sgrnas_sets(
    ky_v11_data.lib, ky_v11_fc, add_controls=True, dataset_name="Yusa_v1.1"
)


# Broad DepMap19Q2 - Avana

avana_data = CRISPRDataSet("Avana")
avana_fc = (
    avana_data.counts.remove_low_counts(avana_data.plasmids)
    .norm_rpm()
    .foldchange(avana_data.plasmids)
)
avana_sgrna_set = define_sgrnas_sets(
    avana_data.lib, avana_fc, add_controls=True, dataset_name="Avana"
)


# sgRNAs JACKS efficiency scores

ky_v11_jacks = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0.csv", index_col=0)
avana_jacks = pd.read_csv(f"{dpath}/jacks/Avana_v1.csv", index_col=0)


# Kolmogorov-Smirnov statistic guide comparison

ky_v11_ks = estimate_ks(ky_v11_fc, ky_v11_sgrna_set["nontargeting"]["fc"])
ky_v11_ks["Gene"] = ky_v11_data.lib.reindex(ky_v11_ks.index)["Gene"].values
ky_v11_ks["jacks"] = ky_v11_jacks.reindex(ky_v11_ks.index)["X1"].values

avana_ks = estimate_ks(avana_fc, avana_sgrna_set["nontargeting"]["fc"])
avana_ks["Gene"] = (
    avana_data.lib.drop_duplicates(subset=["sgRNA"], keep=False)
    .reindex(avana_ks.index)["Gene"]
    .values
)
avana_ks["jacks"] = avana_jacks.reindex(avana_ks.index)["X1"].values


# Libraries aggregated ks scores

ks_genes = pd.concat(
    [
        ky_v11_ks.groupby("Gene")["ks_control"].mean().rename("ky_v11"),
        avana_ks.groupby("Gene")["ks_control"].mean().rename("avana"),
    ],
    axis=1,
    sort=False,
)

ks_genes.to_excel(f"{rpath}/Libraries_KS_scores.xlsx", index=False)


# Plot

plot_df = ks_genes.dropna()

grid = sns.JointGrid("ky_v11", "avana", data=plot_df, space=0)

data, x_e, y_e = np.histogram2d(plot_df["ky_v11"], plot_df["avana"], bins=20)
z_var = interpn(
    (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
    data,
    np.vstack([plot_df["ky_v11"], plot_df["avana"]]).T,
    method="splinef2d",
    bounds_error=False,
)

grid.ax_joint.scatter(
    plot_df["ky_v11"],
    plot_df["avana"],
    c=z_var,
    marker="o",
    edgecolor="",
    cmap="Spectral_r",
    s=5,
    alpha=0.8,
)

rmse = sqrt(mean_squared_error(plot_df["ky_v11"], plot_df["avana"]))
cor, _ = spearmanr(plot_df["ky_v11"], plot_df["avana"])
annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"

grid.ax_joint.text(
    0.95, 0.05, annot_text, fontsize=4, transform=grid.ax_joint.transAxes, ha="right"
)

grid.plot_marginals(
    sns.distplot, kde=False, hist_kws=dict(linewidth=0), color="#6f6dae"
)

lims = [0, 1]
grid.ax_joint.plot(lims, lims, "k-", lw=0.3, zorder=2)
grid.ax_joint.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

grid.ax_joint.set_xlabel("Kosuke v1.1\nGene mean KS non-targeting")
grid.ax_joint.set_ylabel("Avana\nGene mean KS non-targeting")

plt.gcf().set_size_inches(2.5, 2.5)
plt.savefig(f"{rpath}/ks_scatter_kyv11_avana.png", bbox_inches="tight", dpi=600)
plt.close("all")
