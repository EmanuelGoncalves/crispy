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

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from GIPlot import GIPlot
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from minlib.Utils import density_interpolate
from sklearn.metrics import mean_squared_error
from minlib.Utils import project_score_sample_map
from crispy.CRISPRData import CRISPRDataSet, Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# MinLibCas9 library information
#

mlib = Library.load_library("MinLibCas9.csv.gz", set_index=False)
mlib.index = [f"{i}" for i in mlib["WGE_ID"]]


# Assemble raw counts matrix
#

SPATH = pkg_resources.resource_filename("notebooks", "minlib/minlibcas9_screens")

rawcount_files = [i for i in os.listdir(f"{SPATH}") if i.endswith(".tsv.gz")]

rawcounts = pd.concat(
    [
        pd.read_csv(f"{SPATH}/{f}", index_col=0, sep="\t")
        .iloc[:, 1]
        .rename(f.split(".")[0])
        for f in rawcount_files
    ],
    axis=1,
    sort=False,
)
rawcounts = rawcounts.loc[:, ~rawcounts.columns.duplicated(keep="last")]
rawcounts.to_csv(f"{SPATH}/MinLibCas9_rawcounts.csv.gz", compression="gzip")

mlib_dataset = dict(
    name="MinLibCas9_Screens",
    read_counts="MinLibCas9_rawcounts.csv.gz",
    library="MinLibCas9.csv.gz",
    plasmids=["MHG_library_v1"],
)

mlib_data = CRISPRDataSet(mlib_dataset, ddir=SPATH)

mlib_count = mlib_data.counts.remove_low_counts(mlib_data.plasmids)
mlib_fc_sgrna = mlib_count.norm_rpm().foldchange(mlib_data.plasmids)
mlib_fc_gene = mlib_fc_sgrna.groupby(mlib["Approved_Symbol"]).mean()


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")

ky_smap = project_score_sample_map()
ky_smap = ky_smap[[i.startswith("HT29") for i in ky_smap.index]]

ky_counts = ky.counts.remove_low_counts(ky.plasmids)
ky_fc_sgrna = ky_counts.norm_rpm().foldchange(ky.plasmids)[ky_smap.index]
ky_fc_gene = ky_fc_sgrna.groupby(ky.lib["Gene"]).mean()


# Gene fold-change
#

fc_gene = pd.concat([
    ky_fc_gene.add_suffix("_original"),
    mlib_fc_gene.add_suffix("_MinLibCas9"),
], axis=1, sort=False)

lib_pal = dict(original=QCplot.PAL_DBGD[0], MinLibCas9=QCplot.PAL_DBGD[1])
sample_pal = {s: lib_pal[s.split("_")[-1]] for s in fc_gene}
samples = list(fc_gene.columns)


# Fold-changes boxplots
#

plot_df = fc_gene.unstack().rename("fc").reset_index()
plot_df.columns = ["sample", "gene", "fc"]

plt.figure(figsize=(3, 0.25 * fc_gene.shape[1]), dpi=600)
sns.boxplot(
    "fc",
    "sample",
    orient="h",
    data=plot_df,
    order=samples,
    palette=sample_pal,
    saturation=1,
    showcaps=False,
    notch=True,
    boxprops=dict(linewidth=0.3),
    flierprops=QCplot.FLIERPROPS,
)
plt.xlabel("Gene fold-change (log2)")
plt.ylabel("")
plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.savefig(f"{RPATH}/minlibcas9_screens_boxplots_gene_fc.pdf", bbox_inches="tight")
plt.close("all")

# Fold-changes clustermap
#
plot_df = fc_gene.corr(method="spearman")

sns.clustermap(
    plot_df,
    cmap="Spectral",
    annot=True,
    center=0,
    fmt=".2f",
    annot_kws=dict(size=4),
    figsize=(0.5 * len(samples), 0.5 * len(samples)),
    lw=0.05,
    col_colors=pd.Series(sample_pal)[plot_df.columns].rename("Library"),
    row_colors=pd.Series(sample_pal)[plot_df.index].rename("Library"),
    cbar_pos=None,
)

plt.savefig(f"{RPATH}/minlibcas9_screens_clustermap_gene_fc.pdf", bbox_inches="tight")
plt.close("all")


# Recall gene lists
#

gsets_aucs = {}
for n, gset in [
    ("essential", Utils.get_essential_genes()),
    ("non-essential", Utils.get_non_essential_genes()),
]:
    # Aroc
    plt.figure(figsize=(2, 2), dpi=600)
    ax = plt.gca()
    _, stats_ess = QCplot.plot_cumsum_auc(
        fc_gene[samples], gset, palette=sample_pal, legend_prop={"size": 4}, ax=ax
    )
    plt.title(f"{n} recall curve")
    plt.xlabel("Percent-rank of genes")
    plt.ylabel("Cumulative fraction")
    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.savefig(f"{RPATH}/minlibcas9_screens_roccurves_{n}.pdf", bbox_inches="tight")
    plt.close("all")

    # Barplot
    df = pd.Series(stats_ess["auc"])[samples].rename("auc").reset_index()

    plt.figure(figsize=(3, 0.25 * len(samples)), dpi=600)
    sns.barplot(
        "auc",
        "index",
        data=df,
        palette=sample_pal,
        linewidth=0,
        saturation=1,
        orient="h",
    )
    plt.axvline(0.5, ls="-", lw=0.1, alpha=1.0, zorder=0, color="k")
    plt.xlabel("Area under the recall curve")
    plt.ylabel("")
    plt.title(n)
    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.savefig(f"{RPATH}/minlibcas9_screens_roccurves_{n}_barplot.pdf", bbox_inches="tight")
    plt.close("all")

    # Save results
    gsets_aucs[n] = stats_ess["auc"]


# Scatter grid
#

def triu_plot(x, y, color, label, **kwargs):
    z = QCplot.density_interpolate(x, y)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.scatter(x, y, c=z, **kwargs)

    plt.axhline(0, ls=":", lw=0.1, c="#484848", zorder=0)
    plt.axvline(0, ls=":", lw=0.1, c="#484848", zorder=0)

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, ls=":", lw=0.1, c="#484848", zorder=0)


def diag_plot(x, color, label, **kwargs):
    sns.distplot(x, label=label)


grid = sns.PairGrid(fc_gene, height=1.1, despine=False)
grid.map_diag(diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)

for i, j in zip(*np.tril_indices_from(grid.axes, -1)):
    ax = grid.axes[i, j]

    df = pd.concat([fc_gene.iloc[:, i].rename("x"), fc_gene.iloc[:, j].rename("y")], axis=1, sort=False).dropna()
    r, p = spearmanr(df["x"], df["y"])

    ax.annotate(
        f"R={r:.2f}\np={p:.1e}" if p != 0 else f"R={r:.2f}\np<0.0001",
        xy=(0.5, 0.5),
        xycoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )

grid = grid.map_upper(triu_plot, marker="o", edgecolor="", cmap="Spectral_r", s=2)
grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
plt.gcf().set_size_inches(9, 9)
plt.savefig(
    f"{RPATH}/minlibcas9_screens_pairplot_gene_fc.png", bbox_inches="tight", dpi=600
)
plt.close("all")


#
#

plot_df = fc_gene.groupby([i.split("_")[-1] for i in fc_gene.columns], axis=1).mean().dropna()

g = GIPlot.gi_regression("original", "MinLibCas9", plot_df, palette=QCplot.PAL_DBGD)
g.ax_joint.set_xlabel("Original (mean rep. gene FC)")
g.ax_joint.set_ylabel("MinLibCas9 (mean rep. gene FC)")
plt.savefig(
    f"{RPATH}/minlibcas9_screens_pairplot_gene_fc_mean.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
