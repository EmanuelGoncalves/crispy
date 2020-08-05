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
from natsort import natsorted
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from minlib.Utils import replicates_correlation
from crispy.CRISPRData import CRISPRDataSet, Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("KM12_coverage")
ky_counts = ky.counts.remove_low_counts(ky.plasmids)
ky_ss = pd.read_excel(
    f"{DPATH}/crispr_manifests/KM12_coverage_samplesheet.xlsx", index_col="sample"
)
ky_ss["name"] = [f"{c}x ({e})" for e, c in ky_ss[["experiment", "coverage"]].values]
ky_ss["rep_name"] = [f"{n} (Rep{r})" for n, r in ky_ss[["name", "replicate"]].values]

# KY v1.1 library
#

ky_lib = Library.load_library("MasterLib_v1.csv.gz").query("Library == 'KosukeYusa'")
ky_lib = ky_lib[ky_lib.index.isin(ky_counts.index)]
ky_lib_fc = ky_counts.loc[ky_lib.index].norm_rpm().foldchange(ky.plasmids)
ky_lib_gene = ky_lib_fc.groupby(ky_lib["Approved_Symbol"]).mean()
ky_lib_gene_avg = ky_lib_gene.groupby(ky_ss["name"], axis=1).mean()


# Minimal library (top 2)
#

NGUIDES, REMOVE_DISCORDANT = 2, True
ml_lib_name = (
    f"MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz"
)

ml_lib = Library.load_library(ml_lib_name).query("Library == 'KosukeYusa'")
ml_lib = ml_lib[ml_lib.index.isin(ky_counts.index)]
ml_lib = ml_lib.loc[[i for i in ml_lib.index if not i.startswith("CTRL0")]]

ml_lib_fc = ky_counts.loc[ml_lib.index].norm_rpm().foldchange(ky.plasmids)
ml_lib_gene = ml_lib_fc.groupby(ml_lib["Approved_Symbol"]).mean()
ml_lib_gene_avg = ml_lib_gene.groupby(ky_ss["name"], axis=1).mean()


# Comparisons essential AUCs
#

libraries = {
    "Project Score": dict(
        name="Project Score", lib=ky_lib, fc_rep=ky_lib_gene, fc=ky_lib_gene_avg
    ),
    "MinLibCas9": dict(
        name="MinLibCas9", lib=ml_lib, fc_rep=ml_lib_gene, fc=ml_lib_gene_avg
    ),
}

hue_order = ["Project Score", "MinLibCas9"]
pal = {"Project Score": "#e34a33", "MinLibCas9": "#fee8c8"}
order = ["25x (B)", "50x (B)", "75x (B)", "100x (B)", "100x (A)", "500x (A)"]


# Export data
#

minlibcas9 = Library.load_library(ml_lib_name)
kylib = Library.load_library("MasterLib_v1.csv.gz").query("Library == 'KosukeYusa'")

data_export = ky.counts.copy()
data_export.insert(0, "MinLibCas9_guide", data_export.index.isin(minlibcas9.index))
data_export.insert(
    0, "Approved_Symbol", kylib.reindex(data_export.index)["Approved_Symbol"]
)
data_export.to_excel(f"{RPATH}/GuideCoverage_export_data.xlsx")


# Essential genes AROC
#

for l in libraries:
    libraries[l]["aurc"] = pd.Series(
        {
            c: QCplot.aroc_threshold(libraries[l]["fc"][c], fpr_thres=0.2)[0]
            for c in libraries[l]["fc"]
        }
    )


# Replicates correlation
#

for l in libraries:
    libraries[l]["reps"] = replicates_correlation(
        libraries[l]["fc_rep"].rename(columns=ky_ss["name"]), method="spearman"
    )


# Essential genes AURC
#

plot_df = pd.concat(
    [libraries[l]["aurc"].rename(l) for l in libraries], axis=1
).reset_index()
plot_df = pd.melt(plot_df, id_vars="index", value_vars=["Project Score", "MinLibCas9"])

n = libraries["Project Score"]["fc"].shape[1]

fig, ax = plt.subplots(1, 1, figsize=(0.5 * n, 2.0), dpi=600)

sns.barplot(
    "index",
    "value",
    "variable",
    plot_df,
    ax=ax,
    palette=pal,
    linewidth=0,
    order=order,
    hue_order=hue_order,
)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel(None)
ax.set_ylabel("AROC Essential (20% FDR)")

ax.set_ylim(0, 1)

ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_coverage_KM12_ess_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Replicates correlation boxplots
#

plot_df = pd.concat(
    [
        libraries[l]["reps"]
        .query(f"(sample_1 == '{c}') & (sample_2 == '{c}')")
        .assign(library=l)
        .assign(coverage=c)
        for l in libraries
        for c in order
    ]
)

n = len(set(plot_df["coverage"]))

_, ax = plt.subplots(1, 1, figsize=(0.5 * n, 2.0), dpi=600)

sns.barplot(
    "coverage",
    "corr",
    "library",
    plot_df,
    ax=ax,
    ci="sd",
    errwidth=1.0,
    palette=pal,
    linewidth=0,
    order=order,
    hue_order=hue_order,
)

sns.stripplot(
    "coverage",
    "corr",
    "library",
    plot_df,
    ax=ax,
    split=True,
    size=3,
    edgecolor="white",
    color="black",
    linewidth=0.3,
    order=order,
    hue_order=hue_order,
)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel(None)
ax.set_ylabel("Replicates correlation\n(mean Spearman's rho)")

ax.set_ylim(0, 1)

ax.legend(frameon=False)

plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_coverage_KM12_correlation.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

# Scatter grid
#

for l in libraries:
    for dtype in ["fc", "fc_rep"]:

        if dtype == "fc_rep":
            gene_fc = libraries[l][dtype].rename(columns=ky_ss["rep_name"])
        else:
            gene_fc = libraries[l][dtype]

        gene_fc = gene_fc[natsorted(gene_fc)]

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

        def triu_plot_hex(x, y, color, label, **kwargs):
            plt.hexbin(
                x,
                y,
                cmap="Spectral_r",
                gridsize=100,
                mincnt=1,
                bins="log",
                lw=0,
                alpha=1,
            )

            lims = [gene_fc.min().min(), gene_fc.max().max()]
            plt.plot(lims, lims, ls=":", lw=0.1, c="#484848", zorder=0)
            plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
            plt.xlim(lims)
            plt.ylim(lims)

        def diag_plot(x, color, label, **kwargs):
            sns.distplot(x, label=label, color=QCplot.PAL_DBGD[0])

        plot_df = gene_fc.copy()

        grid = sns.PairGrid(plot_df, height=1.1, despine=False)
        grid.map_diag(diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)

        for i, j in zip(*np.tril_indices_from(grid.axes, -1)):
            ax = grid.axes[i, j]

            df = pd.concat(
                [gene_fc.iloc[:, i].rename("x"), gene_fc.iloc[:, j].rename("y")],
                axis=1,
                sort=False,
            ).dropna()
            r, p = spearmanr(df["x"], df["y"])

            ax.annotate(
                f"R={r:.2f}\np={p:.1e}" if p != 0 else f"R={r:.2f}\np<0.0001",
                xy=(0.5, 0.5),
                xycoords=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

        grid = grid.map_upper(triu_plot_hex)
        grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.gcf().set_size_inches(1 * gene_fc.shape[1], 1 * gene_fc.shape[1])
        plt.savefig(
            f"{RPATH}/{l}_{dtype}_reps_scatter.{'png' if dtype == 'fc_rep' else 'pdf'}",
            bbox_inches="tight",
            dpi=600 if dtype == "fc_rep" else None,
        )
        plt.close("all")
