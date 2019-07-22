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
from itertools import groupby
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import PipelineResults
from crispy.CRISPRData import CRISPRDataSet, ReadCounts
from sklearn.metrics import mean_squared_error, matthews_corrcoef
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter, project_score_sample_map


# Project Score

ky = CRISPRDataSet("Yusa_v1.1")

s_map = project_score_sample_map()


# Top2 counts

ky_top2_count = ReadCounts(
    pd.read_csv(
        f"{rpath}/KosukeYusa_v1.1_sgrna_counts_ks_control_min_top2.csv.gz", index_col=0
    )
)
ky_top2_fc = (
    ky_top2_count.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)
)
ky_top2_fc_gene = ky_top2_fc.groupby(ky.lib.loc[ky_top2_fc.index, "Gene"]).mean()
ky_top2_fc_gene_avg = ky_top2_fc_gene.groupby(s_map["model_id"], axis=1).mean()


# All counts

ky_all_count = ReadCounts(
    pd.read_csv(f"{rpath}/KosukeYusa_v1.1_sgrna_counts_all.csv.gz", index_col=0)
)
ky_all_fc = (
    ky_all_count.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)
)
ky_all_fc_gene = ky_all_fc.groupby(ky.lib.loc[ky_all_fc.index, "Gene"]).mean()
ky_all_fc_gene_avg = ky_all_fc_gene.groupby(s_map["model_id"], axis=1).mean()


# sgRNA metrics

ky_metrics = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0)
ky_metrics["top2"] = ky_metrics.index.isin(ky_top2_count.index).astype(int)


# Overlap

dsets = [("Top2", ky_top2_fc_gene), ("All", ky_all_fc_gene)]
dsets_avg = [("Top2", ky_top2_fc_gene_avg), ("All", ky_all_fc_gene_avg)]

genes = list(set(ky_top2_fc_gene.index).intersection(ky_all_fc_gene.index))
samples = list(set(ky_top2_fc_gene).intersection(ky_all_fc_gene))
celllines = list(set(ky_top2_fc_gene_avg).intersection(ky_all_fc_gene_avg))
LOG.info(f"Genes={len(genes)}; Samples={len(samples)}; CellLines={len(celllines)}")


# Calculate AROC 5% FPR

cellline_thres = []

for s in celllines:
    for d, df in dsets_avg:
        fpr_auc, fpr_thres = QCplot.aroc_threshold(df.loc[genes, s])
        cellline_thres.append(dict(sample=s, dtype=d, auc=fpr_auc, fpr_thres=fpr_thres))

cellline_thres = pd.DataFrame(cellline_thres)


# Binarise fold-changes

fc_thres = pd.pivot_table(
    cellline_thres, index="sample", columns="dtype", values="fpr_thres"
)

fc_bin = {}
for d, df in dsets_avg:
    fc_bin[d] = {}
    for s in celllines:
        fc_bin[d][s] = (df.loc[genes, s] < fc_thres.loc[s, d]).astype(int)
    fc_bin[d] = pd.DataFrame(fc_bin[d])


# Comulative number of dependencies

n_iters = 10
ns_thres = np.arange(1, len(celllines), 5)

dep_cumsum = pd.DataFrame(
    [
        dict(
            n_samples=n,
            dtype=t,
            n_deps=fc_bin[t][celllines]
            .sample(n, axis=1)
            .sum(1)
            .replace(0, np.nan)
            .dropna()
            .shape[0],
        )
        for t in fc_bin
        for n in ns_thres
        for i in range(n_iters)
    ]
)


# Plot comulative dependencies

plot_df = pd.pivot_table(dep_cumsum, index="n_samples", columns="dtype", values="n_deps", aggfunc=np.mean)

plt.figure(figsize=(1.75, 1.75), dpi=600)

for i, c in enumerate(plot_df):
    plt.plot(plot_df.index, plot_df[c], label=c, color=CrispyPlot.PAL_DBGD[i])

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Number cell lines")
plt.ylabel("Number dependencies\n5% FPR")

plt.legend(frameon=False)
plt.savefig(f"{rpath}/fpr_bin_cumsum.png", bbox_inches="tight")
plt.close("all")


# AROC and FC threshold scatters

for value in ["auc", "fpr_thres"]:
    plot_df = pd.pivot_table(
        cellline_thres, index="sample", columns="dtype", values=value
    )

    plt.figure(figsize=(2.0, 1.5), dpi=600)
    ax = plt.gca()
    sgrnas_scores_scatter(
        df_ks=plot_df,
        x="All",
        y="Top2",
        annot=True,
        diag_line=True,
        reset_lim=False,
        z_method="gaussian_kde",
        ax=ax,
    )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_xlabel("All sgRNAs")
    ax.set_ylabel("Top 2 sgRNAs")
    ax.set_title(f"{value} 5% FPR")
    plt.savefig(f"{rpath}/fpr_scatter_{value}.png", bbox_inches="tight")
    plt.close("all")


# Number of dependencies scatter

plot_df = pd.concat([fc_bin[c].sum(1).rename(c) for c in fc_bin], axis=1, sort=False)
plot_df["n_sgrnas"] = ky.lib.groupby("Gene")["sgRNA"].count().loc[plot_df.index].values
plot_df["diff"] = plot_df.eval("All - Top2")

plt.figure(figsize=(2.0, 1.5), dpi=600)
ax = plt.gca()
sgrnas_scores_scatter(
    df_ks=plot_df,
    x="All",
    y="Top2",
    annot=True,
    diag_line=True,
    reset_lim=True,
    z_method="gaussian_kde",
    ax=ax,
)
ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel("All sgRNAs")
ax.set_ylabel("Top 2 sgRNAs")
ax.set_title(f"{value} 5% FPR")
plt.savefig(f"{rpath}/fpr_bin_scatter.png", bbox_inches="tight")
plt.close("all")


#

g = "HIST1H4K"

g_metrics = ky_metrics.query(f"Gene == '{g}'").sort_values("ks_control_min")

g_df = ky_all_fc.loc[g_metrics.index].dropna()

row_colors = pd.Series(CrispyPlot.PAL_DBGD)[g_metrics["top2"]]
row_colors.index = g_metrics.index

sns.clustermap(
    g_df,
    cmap="Spectral",
    xticklabels=False,
    figsize=(2, 2),
    row_colors=row_colors,
    center=0,
)

plt.savefig(f"{rpath}/fpr_sgrna_fc_clustermap.png", bbox_inches="tight", dpi=600)
plt.close("all")
