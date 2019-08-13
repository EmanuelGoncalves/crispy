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
from crispy.DataImporter import PipelineResults, Sample
from crispy.CRISPRData import CRISPRDataSet, ReadCounts, Library
from sklearn.metrics import (
    mean_squared_error,
    matthews_corrcoef,
    recall_score,
    precision_score,
)
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter, project_score_sample_map


# Project Score

ky = CRISPRDataSet("Yusa_v1.1")

s_map = project_score_sample_map()


# sgRNAs lib

mlib = (
    Library.load_library("master.csv.gz", set_index=False)
    .query("lib == 'KosukeYusa'")
    .dropna(subset=["Gene"])
    .set_index("sgRNA_ID")
)
mlib = mlib[mlib.index.isin(ky.counts.index)]


# Conditions

DTYPES = ["all", "minimal_top_2", "minimal_top_3"]


# Minimal counts

ky_v11 = {}

for dtype in DTYPES:
    # Read counts
    if dtype == "all":
        counts = ky.counts.copy()

    else:
        counts = ReadCounts(
            pd.read_csv(
                f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{dtype}.csv.gz", index_col=0
            )
        )

    # sgRNA fold-changes
    fc = counts.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)

    # gene fold-changes
    fc_gene = fc.groupby(mlib.loc[fc.index, "Gene"]).mean()

    # gene average fold-changes
    fc_gene_avg = fc_gene.groupby(s_map["model_id"], axis=1).mean()

    # FDR threshold
    fpr = pd.DataFrame(
        {s: QCplot.aroc_threshold(fc_gene_avg[s], fpr_thres=0.01) for s in fc_gene_avg},
        index=["auc", "thres"],
    ).T

    # gene binarised
    fc_gene_avg_bin = (fc_gene_avg < fpr.loc[fc_gene_avg.columns, "thres"].values).astype(int)

    # cumulative dependencies
    n_iters = range(10)
    n_samples = np.arange(1, fc_gene_avg_bin.shape[1], 5)

    cumulative_dep = pd.DataFrame(
        [
            dict(
                n_samples=n,
                n_deps=fc_gene_avg_bin.sample(n, axis=1)
                .sum(1)
                .replace(0, np.nan)
                .dropna()
                .shape[0],
            )
            for n in n_samples
            for i in n_iters
        ]
    )

    # store
    ky_v11[dtype] = dict(
        counts=counts,
        fc=fc,
        fc_gene=fc_gene,
        fc_gene_avg=fc_gene_avg,
        fc_gene_avg_bin=fc_gene_avg_bin,
        lib=mlib.loc[fc.index],
        fpr=fpr,
        cumulative_dep=cumulative_dep,
    )


# sgRNA metrics

ky_metrics = Library.load_library("master.csv.gz", set_index=False)

ky_metrics = ky_metrics.sort_values(
    ["jacks_min", "ks_control_min", "doenchroot"], ascending=[True, True, False]
)

for m in DTYPES:
    ky_metrics[m] = ky_metrics["sgRNA_ID"].isin(ky_v11[m]["fc"].index).astype(int)


# Overlap

dsets = [(k, ky_v11[k]["fc_gene_avg"]) for k in ky_v11]
GENES = set.intersection(*[set(df.index) for _, df in dsets])
SAMPLES = set.intersection(*[set(df.columns) for _, df in dsets])

LOG.info(f"Genes={len(GENES)}; Samples={len(SAMPLES)}")


# Recover dependencies per cell lines

c_recover = []
for c in DTYPES[1:]:
    all_bin = ky_v11["all"]["fc_gene_avg_bin"].loc[GENES, SAMPLES]
    ind_bin = ky_v11[c]["fc_gene_avg_bin"].loc[GENES, SAMPLES]

    ind_metrics = pd.DataFrame(
        [
            dict(
                samples=s,
                dtype=c,
                recall=recall_score(all_bin.iloc[:, i], ind_bin.iloc[:, i]),
                precision=precision_score(all_bin.iloc[:, i], ind_bin.iloc[:, i]),
            )
            for i, s in enumerate(SAMPLES)
        ]
    )

    c_recover.append(ind_metrics)
c_recover = pd.concat(c_recover)


# Plot comulative dependencies

plot_df = pd.concat([ky_v11[c]["cumulative_dep"].assign(dtype=c) for c in DTYPES])
plot_df = plot_df.groupby(["dtype", "n_samples"])["n_deps"].mean().reset_index()

plt.figure(figsize=(1.75, 1.75), dpi=600)

for i, (dtype, df) in enumerate(plot_df.groupby("dtype")):
    plt.plot(df["n_samples"], df["n_deps"], label=dtype, color=CrispyPlot.PAL_DBGD[i])

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Number cell lines")
plt.ylabel("Number dependencies\n1% FPR")

plt.legend(frameon=False)
plt.savefig(f"{rpath}/fpr_bin_cumsum.png", bbox_inches="tight")
plt.close("all")


# AROC and FC threshold scatters

f, axs = plt.subplots(
    2,
    len(DTYPES[1:]),
    sharex="none",
    sharey="row",
    figsize=(1.5 * len(DTYPES[1:]), 3),
    dpi=600,
)

for i, value in enumerate(["auc", "thres"]):
    for j, dtype in enumerate(DTYPES[1:]):
        plot_df = pd.concat(
            [ky_v11[c]["fpr"][value].rename(c) for c in [dtype, "all"]], axis=1
        )

        ax = axs[i][j]
        sgrnas_scores_scatter(
            df_ks=plot_df,
            x=dtype,
            y="all",
            annot=True,
            diag_line=True,
            reset_lim=False,
            z_method="gaussian_kde",
            add_cbar=False,
            ax=ax,
        )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
        ax.set_xlabel(f"{dtype} sgRNAs" if i != 0 else None)
        ax.set_ylabel(f"{value} - All sgRNAs" if j == 0 else None)

plt.savefig(f"{rpath}/fpr_scatter_metrics.png", bbox_inches="tight")
plt.close("all")


# Number of dependencies scatter

plot_df = pd.concat(
    [ky_v11[c]["fc_gene_avg_bin"].loc[GENES, SAMPLES].sum(1).rename(c) for c in ky_v11],
    axis=1,
    sort=False,
)
plot_df.to_csv(f"{rpath}/dep_recap_per_genes.csv")

f, axs = plt.subplots(
    1,
    len(DTYPES[1:]),
    sharex="all",
    sharey="all",
    figsize=(len(DTYPES[1:]) * 1.5, 1.5),
    dpi=600,
)

for i, dtype in enumerate(DTYPES[1:]):
    ax = axs[i]
    sgrnas_scores_scatter(
        df_ks=plot_df,
        x="all",
        y=dtype,
        annot=True,
        diag_line=True,
        reset_lim=True,
        z_method="gaussian_kde",
        add_cbar=False,
        ax=ax,
    )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_xlabel("All sgRNAs")
    ax.set_ylabel(None)
    ax.set_title(f"{dtype} - 1% FPR")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/fpr_bin_scatter.png", bbox_inches="tight")
plt.close("all")


# sgRNA fold-changes clustermap

x = pd.concat([ky_v11[c]["fc_gene_avg_bin"].loc["WRN"].rename(c) for c in DTYPES], axis=1)
x = pd.concat([x, Sample().samplesheet.loc[x.index, "msi_status"]], axis=1).dropna()

x.sort_values(["all", DTYPES[1], "msi_status"], ascending=[False, False, True])

g = "SDK1"

g_metrics = ky_metrics.query(f"Gene == '{g}'").sort_values("ks_control_min")

g_df = ky_v11["all"]["fc"].loc[g_metrics["sgRNA_ID"].values].dropna()

row_colors = pd.Series(CrispyPlot.PAL_DBGD)[g_metrics[DTYPES[1]]]
row_colors.index = g_metrics["sgRNA_ID"].values

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


# Recover scatter

fig, ax1 = plt.subplots(figsize=(2.5, 1.5), dpi=600)

for i, (n, df) in enumerate(c_recover.groupby("dtype")):
    df = df.sort_values("recall", ascending=False).reset_index(drop=True)

    ax1.plot(df.index, df["recall"], c=CrispyPlot.PAL_DBGD[i], label=n)

    for t in [.8, .9]:
        s_t = df[df["recall"] < t].index[0]
        ax1.axvline(s_t, lw=0.1, c=CrispyPlot.PAL_DBGD[i], zorder=0)

        s_t_text = f"{((s_t / df.shape[0]) * 100):.1f}%"
        ax1.text(
            s_t,
            plt.ylim()[0],
            s_t_text,
            fontsize=3,
            c=CrispyPlot.PAL_DBGD[i],
        )

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.xlabel("Number of cell lines")
plt.ylabel("Recall dependencies\n1% FDR")

plt.legend(frameon=False, prop={"size": 3}).get_title().set_fontsize("3")

plt.title("All vs Minimal")
plt.savefig(f"{rpath}/fpr_recover_scatter.png", bbox_inches="tight")
plt.close("all")
