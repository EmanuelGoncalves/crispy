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
from math import sqrt
from minlib import rpath, LOG
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error
from crispy.CRISPRData import CRISPRDataSet, Library


# CRISPR-Cas9 screens in 2 colorectal cancer organoids

org_data = CRISPRDataSet("Organoids")
org_count = org_data.counts.remove_low_counts(org_data.plasmids)


# Minimal library

min_lib = Library.load_library("minimal_top_2.csv.gz")
min_lib = min_lib[min_lib.index.isin(org_count.index)]


# Master library

master_lib = (
    Library.load_library("master.csv.gz")
    .query("lib == 'KosukeYusa'")
    .dropna(subset=["Gene"])
)
master_lib = master_lib[master_lib.index.isin(org_count.index)]
master_lib = master_lib[master_lib["Gene"].isin(min_lib["Gene"])]


# Gene fold-changes

org_fc_gene = {}
for n, library in [("all", master_lib), ("minimal_top_2", min_lib)]:
    fc = org_count.loc[library.index].norm_rpm().foldchange(org_data.plasmids)
    fc = fc.loc[library.index].groupby(library["Gene"]).mean()
    org_fc_gene[n] = fc.round(5)


# Essential/non-essential AROC and AURC

metrics_arocs = []
for m in org_fc_gene:
    for s in org_fc_gene[m]:
        LOG.info(f"Metric={m}; Organoids={s}")

        metrics_arocs.append(dict(
            metric=m,
            sample=s,
            aroc=QCplot.pr_curve(org_fc_gene[m][s]),
            aurc=QCplot.recall_curve(org_fc_gene[m][s])[2],
            ess_recall=QCplot.recall_curve(org_fc_gene[m][s])[2],
            average_precision=QCplot.precision_recall_curve(org_fc_gene[m][s])[0]

        ))
metrics_arocs = pd.DataFrame(metrics_arocs)


# Plot essential genes recall

metrics = ["aroc", "aurc", "average_precision", "ess_recall"]

pal = dict(all="#e34a33", minimal_top_2="#fee8c8")

f, axs = plt.subplots(1, len(metrics), sharex="all", sharey="all", figsize=(1.5 * len(metrics), 1.5), dpi=600)

for i, metric in enumerate(metrics):
    ax = axs[i]

    sns.barplot("sample", metric, "metric", metrics_arocs, ax=ax, palette=pal, linewidth=0)

    ax.set_xlabel("")
    ax.set_ylabel("Area under" if i == 0 else "")
    ax.set_title(metric)
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_ylim(0.5, 1)

    ax.get_legend().remove()

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 6}, title="sgRNAs")
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/organoids_ess_barplot.pdf", bbox_inches="tight")
plt.close("all")


# Gene fold-change scatter

x_min = min([org_fc_gene[m].min().min() for m in org_fc_gene])
x_max = max([org_fc_gene[m].max().max() for m in org_fc_gene])

f, axs = plt.subplots(3, 1, sharex="all", sharey="all", figsize=(1.5, 4.5), dpi=600)

for i, s in enumerate(org_fc_gene["minimal_top_2"]):
    ax = axs[i]

    y_var = org_fc_gene["all"][s]
    x_var = org_fc_gene["minimal_top_2"][s]

    data, x_e, y_e = np.histogram2d(x_var, y_var, bins=20)
    z_var = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x_var, y_var]).T,
        method="splinef2d",
        bounds_error=False,
    )

    ax.scatter(
        x_var, y_var,
        c=z_var,
        marker="o",
        edgecolor="",
        s=3,
        cmap="Spectral_r",
        alpha=0.3,
    )

    ax.set_xlabel(f"Minimal\nsgRNA fold-change" if i == 2 else "")
    ax.set_ylabel(f"All\nsgRNA fold-change")

    lims = [x_max, x_min]
    ax.plot(lims, lims, "k-", lw=0.3, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    ax_inv = ax.twinx()
    ax_inv.set_ylabel(s)
    ax_inv.get_yaxis().set_ticks([])

    rmse = sqrt(mean_squared_error(x_var, y_var))
    cor, _ = spearmanr(x_var, y_var)
    annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"
    ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/organoids_gene_fc_scatter.png", bbox_inches="tight")
plt.close("all")
