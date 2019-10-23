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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from minlib.Utils import density_interpolate
from sklearn.metrics import mean_squared_error
from crispy.CRISPRData import CRISPRDataSet, Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# CRISPR-Cas9 screens in 2 colorectal cancer organoids
#

org_data = CRISPRDataSet("Organoids")
org_count = org_data.counts.remove_low_counts(org_data.plasmids)


# Libraries
#

NGUIDES, REMOVE_DISCORDANT = 2, True
ml_lib_name = (
    f"MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz"
)
ml_lib = Library.load_library(ml_lib_name).query("Library == 'KosukeYusa'")
ml_lib = ml_lib.loc[[i for i in ml_lib.index if not i.startswith("CTRL0")]]

libraries = dict(
    All=dict(
        name="All",
        lib=Library.load_library("MasterLib_v1.csv.gz").query(
            "Library == 'KosukeYusa'"
        ),
    ),
    Minimal=dict(name="Minimal", lib=ml_lib),
)


# Gene fold-changes
#

for ltype in libraries:
    lib = libraries[ltype]["lib"]

    counts = org_count[org_count.index.isin(lib.index)]

    fc = counts.norm_rpm().foldchange(org_data.plasmids)
    fc = fc.groupby([c.split("_")[0] for c in fc], axis=1).mean()
    fc = fc.groupby(lib["Approved_Symbol"]).mean()

    libraries[ltype]["counts"] = counts
    libraries[ltype]["fc"] = fc


# Genes overlap
#

genes = set(libraries["All"]["fc"].index).intersection(libraries["Minimal"]["fc"].index)
LOG.info(f"Genes={len(genes)}")


# Essential/non-essential AROC and AURC
#

metrics_arocs = []
for ltype in libraries:
    for s in libraries[ltype]["fc"]:
        LOG.info(f"Library={ltype}; Organoid={s}")

        metrics_arocs.append(
            dict(
                library=ltype,
                sample=s,
                aroc=QCplot.aroc_threshold(libraries[ltype]["fc"][s], fpr_thres=0.2)[0],
            )
        )
metrics_arocs = pd.DataFrame(metrics_arocs)


# Plot essential genes recall

pal = dict(All="#e34a33", Minimal="#fee8c8")

n = libraries["All"]["fc"].shape[1]

fig, ax = plt.subplots(1, 1, figsize=(0.6 * n, 2.0), dpi=600)

sns.barplot("sample", "aroc", "library", metrics_arocs, ax=ax, palette=pal, linewidth=0)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.set_xlabel(None)
ax.set_ylabel("AROC Essential (20% FDR)")

ax.set_ylim(0.5, 1)
plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig(f"{RPATH}/organoids_ess_barplot.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


# Gene fold-change scatter
#

x_min = min([libraries[m]["fc"].min().min() for m in libraries])
x_max = max([libraries[m]["fc"].max().max() for m in libraries])

n = libraries["All"]["fc"].shape[1]

f, axs = plt.subplots(1, n, sharex="all", sharey="all", figsize=(2 * n, 2), dpi=600)

for i, s in enumerate(libraries["All"]["fc"]):
    ax = axs[i]

    x_var = libraries["All"]["fc"].loc[genes, s]
    y_var = libraries["Minimal"]["fc"].loc[genes, s]
    z_var = density_interpolate(x_var, y_var)

    ax.scatter(
        x_var,
        y_var,
        c=z_var,
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=3,
        alpha=0.7,
    )

    ax.set_xlabel(f"Kosuke Yusa V1.1 (5 sgRNAs/Gene)\nGene fold-change")
    ax.set_ylabel(f"MinLibCas9 (2 sgRNAs/Gene)\nGene fold-change" if i == 0 else None)
    ax.set_title(s)

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    lims = [x_max, x_min]
    ax.plot(lims, lims, "k-", lw=0.3, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)

    rmse = sqrt(mean_squared_error(x_var, y_var))
    cor, _ = spearmanr(x_var, y_var)
    annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"
    ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(
    f"{RPATH}/organoids_gene_fc_scatter.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")
