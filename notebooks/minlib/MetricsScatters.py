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
from scipy.stats import spearmanr
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib.Utils import define_sgrnas_sets, density_interpolate


RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Master library (KosukeYusa v1.1 + Avana + Brunello)
#

master_lib = Library.load_library("MasterLib_v1.csv.gz", set_index=False)
master_lib = master_lib.query("Library == 'KosukeYusa'")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")

ky_counts = ky.counts.remove_low_counts(ky.plasmids)

ky_fc = ky_counts.norm_rpm().norm_rpm().foldchange(ky.plasmids)

ky_gsets = define_sgrnas_sets(ky.lib, ky_fc, add_controls=True)

ky_gmetrics = pd.concat(
    [
        ky_fc.median(1).rename("Median"),
        master_lib.set_index("sgRNA_ID")[["JACKS", "RuleSet2", "FORECAST", "KS"]],
    ],
    axis=1,
    sort=False,
)


# sgRNA metrics scatter plots
#

x_vars, y_var = ["KS", "JACKS", "RuleSet2", "FORECAST"], "Median"

f, axs = plt.subplots(
    1, len(x_vars), sharex="none", sharey="row", figsize=(2 * len(x_vars), 2)
)

for i, x_var in enumerate(x_vars):
    ax = axs[i]

    plot_df = ky_gmetrics[[x_var, y_var]].dropna()
    plot_df["z_var"] = density_interpolate(plot_df[x_var], plot_df[y_var])
    plot_df = plot_df.sort_values("z_var")

    ax.hexbin(
        plot_df[x_var],
        plot_df[y_var],
        cmap="Spectral_r",
        gridsize=50,
        mincnt=1,
        bins="log",
        lw=0,
    )

    cor, _ = spearmanr(plot_df[[x_var, y_var]].values)
    annot_text = f"Spearman's R={cor:.2g}"
    ax.text(0.95, 0.05, annot_text, fontsize=5, transform=ax.transAxes, ha="right")

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    for g in ky_gsets:
        ax.axhline(
            ky_gsets[g]["fc"].median(),
            ls="-",
            lw=0.5,
            c=ky_gsets[g]["color"],
            zorder=0,
            label=g,
        )

    ax.set_xlabel(f"sgRNA {x_var}")
    ax.set_ylabel("sgRNA median fold-change" if i == 0 else None)

    if i == 0:
        ax.legend(prop={"size": 5}, frameon=False, loc=3)

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{RPATH}/ky_v11_guides_metrics_scatter.pdf", bbox_inches="tight")
plt.close("all")
