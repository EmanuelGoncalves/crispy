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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
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

libraries = dict(
    All=dict(name="All", lib=ky_lib, fc_rep=ky_lib_gene, fc=ky_lib_gene_avg),
    Minimal=dict(name="Minimal", lib=ml_lib, fc_rep=ml_lib_gene, fc=ml_lib_gene_avg),
)

hue_order = ["All", "Minimal"]
pal = dict(All="#e34a33", Minimal="#fee8c8")
order = ["25x (B)", "50x (B)", "75x (B)", "100x (B)", "100x (A)", "500x (A)"]


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
        libraries[l]["fc_rep"].rename(columns=ky_ss["name"])
    )


# Essential genes AURC
#

plot_df = pd.concat(
    [libraries[l]["aurc"].rename(l) for l in libraries], axis=1
).reset_index()
plot_df = pd.melt(plot_df, id_vars="index", value_vars=["All", "Minimal"])

n = libraries["All"]["fc"].shape[1]

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

ax.set_ylim(0.5, 1)

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

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel(None)
ax.set_ylabel("Replicates correlation\n(mean Pearson)")

ax.set_ylim(0.5, 1)

ax.legend(frameon=False)

plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_coverage_KM12_correlation.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
