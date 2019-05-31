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
from crispy.QCPlot import QCplot
from itertools import combinations
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from C50K import clib_palette, LOG, rpath, dpath
from crispy.CRISPRData import CRISPRDataSet, Library


# KY v1.1 library

ky_v11_lib = Library.load_library("Yusa_v1.1.csv.gz")
sabatini_lib = Library.load_library("Sabatini_Lander_v3.csv.gz")
avana_lib = Library.load_library("Avana_v1.csv.gz")


# KY v1.0 & v1.1 sgRNA metrics

lib_metrics = pd.read_excel(
    f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0
).sort_values("ks_control_min")


# Avana sgRNA metrics

avana_metrics = pd.read_excel(
    f"{rpath}/Avana_sgRNA_metrics.xlsx", index_col=0
).sort_values("ks_control_min")


# JACKS v1.1
jacks_v11 = pd.read_csv(f"{dpath}/jacks/Yusa_v1.1.csv", index_col=0)
lib_metrics["jacks_v11"] = jacks_v11.reindex(lib_metrics.index)["X1"].values
lib_metrics["jacks_v11_min"] = abs(lib_metrics["jacks_v11"] - 1)


# Top genes

ks_top2 = lib_metrics.sort_values("ks_control_min").groupby("Gene").head(2)
ks_top3 = lib_metrics.sort_values("ks_control_min").groupby("Gene").head(3)


# Plot top 2 vs top 3 mean gene KS score

plot_df = pd.concat(
    [
        ks_top2.groupby("Gene")["ks_control"].mean().rename("top2"),
        ks_top3.groupby("Gene")["ks_control"].mean().rename("top3"),
    ],
    axis=1,
    sort=False,
)

x_min, x_max = plot_df.min().min(), plot_df.max().max()

plt.figure(figsize=(1.75, 1.75), dpi=600)

data, x_e, y_e = np.histogram2d(plot_df["top2"], plot_df["top3"], bins=20)
z_var = interpn(
    (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
    data,
    np.vstack([plot_df["top2"], plot_df["top3"]]).T,
    method="splinef2d",
    bounds_error=False,
)

plt.scatter(
    plot_df["top2"],
    plot_df["top3"],
    c=z_var,
    marker="o",
    edgecolor="",
    s=3,
    cmap="Spectral_r",
    alpha=0.3,
)

plt.xlabel("KS score mean gene Top 2")
plt.ylabel("KS score mean gene Top 3")
plt.title("KS gene rank")

lims = [x_min, x_max]
plt.plot(lims, lims, "k-", lw=0.3, zorder=2)
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(f"{rpath}/minimal_lib_ks_top2_top3_scatter.png", bbox_inches="tight")
plt.close("all")


#

minimal_lib = []
for g in set(lib_metrics["Gene"]):
    LOG.info(f"Gene={g}")

    g_metric = lib_metrics.query(f"Gene == '{g}'")

    # Remove guides with potential off-target and non-targetting effects
    g_guides = g_metric[g_metric["jacks_v11_min"] < 1.5].head(2)
    g_guides = g_guides.assign(info="pass")

    if len(g_guides) == 0:
        g_guides = g_metric.head(2)
        g_guides = g_guides.assign(info="JACKS 2 sgRNAs failed")
        LOG.warning("JACKS 2 sgRNAs failed")

    elif len(g_guides) == 1:
        g_guides = pd.concat(
            [g_guides, g_metric[g_metric["jacks_v11_min"] < 2.5].head(1)], sort=False
        )
        g_guides = g_guides.assign(info="JACKS 1 sgRNAs failed")
        LOG.warning("JACKS 1 sgRNAs failed")

    minimal_lib.append(g_guides)

minimal_lib = pd.concat(minimal_lib)
