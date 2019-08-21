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
import pkg_resources
import matplotlib.pyplot as plt
from crispy.CRISPRData import CRISPRDataSet
from minlib.Utils import project_score_sample_map, density_interpolate, downsample_sgrnas


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# Downsample number of sgRNAs per gene
#

ds_scores = downsample_sgrnas(
    ky_counts, ky.lib, ky_smap, [1, 2, 3, 4, 5, 100], n_iters=10
)
ds_scores.to_excel(f"{RPATH}/YusaDownsample_AROCs.xlsx", index=False)


# Plot
#

x_features, y_feature = np.arange(1, 6), 100

plot_df = ds_scores.groupby(["sample", "n_guides"])["aroc"].mean().reset_index()

xy_min, xy_max = plot_df["aroc"].min(), plot_df["aroc"].max()

y_var = (
    plot_df.query(f"n_guides == {y_feature}").set_index("sample")["aroc"].rename("y")
)

f, axs = plt.subplots(
    1, len(x_features), sharex="all", sharey="all", figsize=(1.5 * len(x_features), 1.5)
)

for i, (n, df) in enumerate(
    plot_df.query(f"n_guides != {y_feature}").groupby("n_guides")
):
    x_var = df.set_index("sample")["aroc"].rename("x")
    z_var = density_interpolate(x_var, y_var)

    axs[i].scatter(
        x_var,
        y_var,
        c=z_var,
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=3,
        alpha=0.7,
    )

    axs[i].grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

    lims = [xy_max, xy_min]
    axs[i].plot(lims, lims, "k-", lw=0.3, zorder=0)

    axs[i].set_xlabel(f"{n} sgRNA (per gene)" if i == 0 else f"{n} sgRNAs (per gene)")
    axs[i].set_ylabel(f"All sgRNAs" if i == 0 else None)

plt.suptitle(f"Essential genes AROC (1% FDR)", y=1.02)
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{RPATH}/YusaDownsample_AROCs_scatter.pdf", bbox_inches="tight")
plt.close("all")
