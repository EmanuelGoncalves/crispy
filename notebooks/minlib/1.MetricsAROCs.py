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
import matplotlib.pyplot as plt
from minlib import rpath
from scipy.interpolate import interpn


# Metrics essential/non-essential AROC

ky_v11_arocs = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_aroc.xlsx")


# Plot AROC

nguides_thres = 5

x_feature = "random"
y_features = ["ks_control_min", "ks_control", "doenchroot", "doenchroot_min", "jacks_min", "median_fc"]

f, axs = plt.subplots(
    len(y_features),
    nguides_thres,
    sharex="all",
    sharey="all",
    figsize=(6, len(y_features)),
    dpi=600,
)

x_min, x_max = ky_v11_arocs["auc"].min(), ky_v11_arocs["auc"].max()

for i in range(nguides_thres):
    plot_df = ky_v11_arocs.query(f"n_guides == {(i + 1)}")

    x_var = (
        plot_df.query(f"dtype == '{x_feature}'").set_index("index")["auc"].rename("x")
    )
    y_vars = pd.concat(
        [
            plot_df.query(f"dtype == '{f}'")
            .set_index("index")["auc"]
            .rename(f"y{i + 1}")
            for i, f in enumerate(y_features)
        ],
        axis=1,
    )
    plot_df = pd.concat([x_var, y_vars], axis=1)

    for j in range(len(y_features)):
        data, x_e, y_e = np.histogram2d(plot_df["x"], plot_df[f"y{j + 1}"], bins=20)
        z_var = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([plot_df["x"], plot_df[f"y{j + 1}"]]).T,
            method="splinef2d",
            bounds_error=False,
        )

        axs[j][i].scatter(
            plot_df["x"],
            plot_df[f"y{j + 1}"],
            c=z_var,
            marker="o",
            edgecolor="",
            cmap="Spectral_r",
            s=3,
            alpha=0.7,
        )

        lims = [x_max, x_min]
        axs[j][i].plot(lims, lims, "k-", lw=0.3, zorder=0)

    axs[0][i].set_title(f"#(guides) = {i + 1}")

    if i == 0:
        for l_index, l in enumerate(y_features):
            axs[l_index][i].set_ylabel(f"{l}\nAROC")

    axs[len(y_features) - 1][i].set_xlabel(f"{x_feature}\nAROC")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_ks_guides_benchmark_scatter.png", bbox_inches="tight")
plt.show()
