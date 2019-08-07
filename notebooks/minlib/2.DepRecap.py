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
from sklearn.metrics import mean_squared_error, matthews_corrcoef
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter, project_score_sample_map


# Project Score

ky = CRISPRDataSet("Yusa_v1.1")

s_map = project_score_sample_map()


# Minimal counts

ky_v11 = {}

for n in ["minimal_top_2", "minimal_top_3", "all"]:
    if n == "all":
        counts = ky.counts.copy()

    else:
        counts = ReadCounts(
            pd.read_csv(f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{n}.csv.gz", index_col=0)
        )

    fc = counts.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)

    fc_gene = fc.groupby(ky.lib.loc[fc.index, "Gene"]).mean()

    fc_gene_avg = fc_gene.groupby(s_map["model_id"], axis=1).mean()

    ky_v11[n] = dict(counts=counts, fc=fc, fc_gene=fc_gene, fc_gene_avg=fc_gene_avg, lib=ky.lib.loc[fc.index])


# sgRNA metrics

ky_metrics = Library.load_library("master.csv.gz", set_index=False)
ky_metrics = ky_metrics.sort_values(
    ["ks_control_min", "jacks_min", "doenchroot"], ascending=[True, True, False]
)
ky_metrics["top2"] = ky_metrics["sgRNA_ID"].isin(ky_v11["minimal_top_2"]["fc"].index).astype(
    int
)
ky_metrics["top3"] = ky_metrics["sgRNA_ID"].isin(ky_v11["minimal_top_3"]["fc"].index).astype(
    int
)


# Overlap

dsets = [(k, ky_v11[k]["fc_gene_avg"]) for k in ky_v11]

genes_top2 = ky_v11["minimal_top_2"]["lib"]["Gene"].value_counts()
genes = set.intersection(*[set(df.index) for _, df in dsets])
genes = genes.intersection(genes_top2[genes_top2 == 2].index)

samples = set.intersection(*[set(df.columns) for _, df in dsets])

LOG.info(f"Genes={len(genes)}; Samples={len(samples)}")


# Calculate AROC 1% FPR

samples_fpr_thres = []

for s in samples:
    for d, df in dsets:
        fpr_auc, fpr_thres = QCplot.aroc_threshold(df.loc[genes, s], fpr_thres=0.01)
        samples_fpr_thres.append(
            dict(sample=s, dtype=d, auc=fpr_auc, fpr_thres=fpr_thres)
        )

samples_fpr_thres = pd.DataFrame(samples_fpr_thres)


# Binarise fold-changes

fc_thres = pd.pivot_table(
    samples_fpr_thres, index="sample", columns="dtype", values="fpr_thres"
)

fc_bin = {}
for d, df in dsets:
    fc_bin[d] = {}
    for s in samples:
        fc_bin[d][s] = (df.loc[genes, s] < fc_thres.loc[s, d]).astype(int)
    fc_bin[d] = pd.DataFrame(fc_bin[d])


# Recover dependencies per cell lines

c_recover = []
for s in samples:
    s_all = fc_bin["all"][s]
    s_all = s_all[s_all == 1]
    s_all = set(s_all.index)

    for t in ["minimal_top_2", "minimal_top_3"]:
        s_top = fc_bin[t][s]
        s_top = s_top[s_top == 1]
        s_top = set(s_top.index)

        c_recover.append(
            dict(
                sample=s,
                minimal=t,
                union=len(s_all.union(s_top)),
                intersection=len(set(s_all).intersection(s_top)),
                all_only=len(set(s_all) - set(s_top)),
                top_only=len(set(s_top) - set(s_all)),
            )
        )

c_recover = pd.DataFrame(c_recover)
c_recover["recover"] = c_recover.eval("intersection / (intersection + all_only)") * 100


# #
#
# dep_lost = []
# for s in samples:
#     s_all = fc_bin["all"][s]
#     s_all = s_all[s_all == 1]
#     s_all = set(s_all.index)
#
#     s_top = fc_bin["minimal_top_2"][s]
#     s_top = s_top[s_top == 1]
#     s_top = set(s_top.index)
#
#     dep_lost += list(s_all - s_top)
#
# dep_lost = pd.Series(dep_lost).value_counts()
# dep_lost.to_clipboard()


# Comulative number of dependencies

n_iters = 10
ns_thres = np.arange(1, len(samples), 5)

dep_cumsum = pd.DataFrame(
    [
        dict(
            n_samples=n,
            dtype=t,
            n_deps=fc_bin[t][samples]
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

plot_df = pd.pivot_table(
    dep_cumsum, index="n_samples", columns="dtype", values="n_deps", aggfunc=np.mean
)

plt.figure(figsize=(1.75, 1.75), dpi=600)

for i, c in enumerate(plot_df):
    plt.plot(plot_df.index, plot_df[c], label=c, color=CrispyPlot.PAL_DBGD[i])

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Number cell lines")
plt.ylabel("Number dependencies\n1% FPR")

plt.legend(frameon=False)
plt.savefig(f"{rpath}/fpr_bin_cumsum.png", bbox_inches="tight")
plt.close("all")


# AROC and FC threshold scatters

f, axs = plt.subplots(2, 2, sharex="none", sharey="row", figsize=(3, 3), dpi=600)

for i, value in enumerate(["auc", "fpr_thres"]):
    for j, dtype in enumerate(["minimal_top_2", "minimal_top_3"]):
        plot_df = pd.pivot_table(
            samples_fpr_thres[samples_fpr_thres["dtype"].isin(["all", dtype])],
            index="sample",
            columns="dtype",
            values=value,
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

plot_df = pd.concat([fc_bin[c].sum(1).rename(c) for c in fc_bin], axis=1, sort=False)

f, axs = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(3, 1.5), dpi=600)

for i, dtype in enumerate(["minimal_top_2", "minimal_top_3"]):
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

x = pd.concat([fc_bin[c].loc["WRN"].rename(c) for c in fc_bin], axis=1)
x = pd.concat([x, Sample().samplesheet.loc[x.index, "msi_status"]], axis=1).dropna()

x.sort_values(["all", "minimal_top_2", "msi_status"], ascending=[False, False, True])

g = "WRN"

g_metrics = ky_metrics.query(f"Gene == '{g}'").sort_values("ks_control_min")

g_df = ky_v11["all"]["fc"].loc[g_metrics["sgRNA_ID"].values].dropna()

row_colors = pd.Series(CrispyPlot.PAL_DBGD)[g_metrics["top2"]]
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

plt.figure(figsize=(2.5, 1.5), dpi=600)

for i, (n, df) in enumerate(c_recover.groupby("minimal")):
    df = df.sort_values("recover", ascending=False).reset_index(drop=True)
    plt.scatter(
        df.index, df["recover"], c=CrispyPlot.PAL_DBGD[1 - i], s=2, lw=0, label=n
    )

    for t in [80, 90]:
        s_t = df[df["recover"] < t].index[0]
        plt.axvline(s_t, lw=0.1, c=CrispyPlot.PAL_DBGD[1 - i], zorder=0)

        s_t_text = f"{((s_t / df.shape[0]) * 100):.1f}%"
        plt.text(
            s_t,
            plt.ylim()[0] + 1 + (i * 2),
            s_t_text,
            fontsize=3,
            c=CrispyPlot.PAL_DBGD[1 - i],
        )

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.xlabel("Number of cell lines")
plt.ylabel("Recovered dependencies\n1% FDR")

plt.legend(frameon=False, prop={"size": 3}).get_title().set_fontsize("3")

plt.title("All vs Minimal")
plt.savefig(f"{rpath}/fpr_recover_scatter.png", bbox_inches="tight")
plt.close("all")
