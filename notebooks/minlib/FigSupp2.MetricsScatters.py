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
from crispy.CrispyPlot import CrispyPlot
from minlib.Utils import define_sgrnas_sets
from crispy.CRISPRData import CRISPRDataSet, Library


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


def triu_plot(x, y, color, label, **kwargs):
    df = pd.DataFrame(dict(x=x, y=y)).dropna()
    ax = plt.gca()
    ax.hexbin(
        df["x"],
        df["y"],
        cmap="Spectral_r",
        gridsize=50,
        mincnt=1,
        bins="log",
        lw=0,
    )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)


def diag_plot(x, color, label, **kwargs):
    sns.distplot(x[~np.isnan(x)], label=label, color=CrispyPlot.PAL_DBGD[0], kde=False)


order = ["Median", "KS", "JACKS", "RuleSet2", "FORECAST"]

genes = {
    "All": ky_gmetrics.index,
    "Essential": ky_gsets["essential"]["sgrnas"],
    "Non-essential": ky_gsets["nonessential"]["sgrnas"],
}

for dtype in genes:
    plot_df = ky_gmetrics.reindex(genes[dtype])

    grid = sns.PairGrid(plot_df[order], height=1.1, despine=False, diag_sharey=False)

    for j, i in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (1, 2), (2, 3), (2, 4), (3, 4)]:
        ax = grid.axes[i, j]
        r, p = spearmanr(plot_df[order[i]], plot_df[order[j]], nan_policy="omit")
        ax.annotate(
            f"R={r:.2f}\np={p:.1e}" if p != 0 else f"R={r:.2f}\np<0.0001",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    grid.map_diag(diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)
    grid.map_upper(triu_plot, marker="o", edgecolor="", cmap="Spectral_r", s=2)

    plt.title(dtype)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.gcf().set_size_inches(1.5 * len(order), 1.5 * len(order))
    plt.savefig(f"{RPATH}/ky_v11_guides_metrics_scatter_{dtype}.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")
