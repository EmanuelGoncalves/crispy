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
from DataImporter import Sample
from crispy.QCPlot import QCplot
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import recall_score, precision_score
from crispy.CRISPRData import CRISPRDataSet, Library, ReadCounts
from minlib.Utils import project_score_sample_map, density_interpolate


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# KY v1.1 library
#

ky_lib = Library.load_library("MasterLib_v1.csv.gz").query("Library == 'KosukeYusa'")
ky_lib = ky_lib[ky_lib.index.isin(ky_counts.index)]


# Minimal library (top 2)
#

ml_lib = Library.load_library("MinimalLib_top2.csv.gz").query("Library == 'KosukeYusa'")
ml_lib = ml_lib[ml_lib.index.isin(ky_counts.index)]


# Genes overlap
#

genes = set(ky_lib["Approved_Symbol"]).intersection(ml_lib["Approved_Symbol"])
LOG.info(f"Genes={len(genes)}")


# Comparisons
#

FDR_THRES = 0.01

libraries = dict(
    All=dict(name="All", lib=ky_lib),
    Minimal=dict(name="Minimal", lib=ml_lib),
)


# Minimal counts
#

for ltype in libraries:
    lib = libraries[ltype]["lib"].copy()

    # sgRNA fold-changes
    fc = ky_counts.copy().loc[lib.index].norm_rpm().foldchange(ky.plasmids)

    # Gene fold-changes
    fc_gene = fc.groupby(lib["Approved_Symbol"]).mean()

    # Gene average fold-changes
    fc_gene_avg = fc_gene.groupby(ky_smap["model_id"], axis=1).mean()

    # Gene average scaled fold-changes
    fc_gene_avg_scl = ReadCounts(fc_gene_avg).scale()

    # FDR threshold
    fpr = pd.DataFrame(
        {
            s: QCplot.aroc_threshold(fc_gene_avg[s], fpr_thres=FDR_THRES)
            for s in fc_gene_avg
        },
        index=["auc", "thres"],
    ).T
    fpr["auc"] = [QCplot.aroc_threshold(fc_gene_avg[s], fpr_thres=.2)[0] for s in fpr.index]

    # Gene binarised
    fc_gene_avg_bin = (fc_gene_avg[fpr.index] < fpr["thres"]).astype(int)

    # Cumulative dependencies
    n_samples = np.arange(5, fc_gene_avg_bin.shape[1], 5)

    cumulative_dep = []
    for i in range(10):
        for n in n_samples:
            genes_dep = fc_gene_avg_bin.sample(n, axis=1).sum(1)
            cumulative_dep.append(dict(
                n_samples=n,
                n_deps=sum(genes_dep >= 3),
            ))
    cumulative_dep = pd.DataFrame(cumulative_dep)

    # Store
    libraries[ltype]["fc"] = fc
    libraries[ltype]["fc_gene"] = fc_gene
    libraries[ltype]["fc_gene_avg"] = fc_gene_avg
    libraries[ltype]["fc_gene_avg_scl"] = fc_gene_avg_scl
    libraries[ltype]["fc_gene_avg_bin"] = fc_gene_avg_bin
    libraries[ltype]["fpr"] = fpr
    libraries[ltype]["cumulative_dep"] = cumulative_dep


# Recovered dependencies per cell line
#

deprecovered = pd.DataFrame(
    [
        dict(
            samples=s,
            recall=recall_score(
                libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
                libraries["Minimal"]["fc_gene_avg_bin"].loc[genes, s],
            ),
            precision=precision_score(
                libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
                libraries["Minimal"]["fc_gene_avg_bin"].loc[genes, s],
            ),
            spearman=spearmanr(
                libraries["All"]["fc_gene_avg"].loc[genes, s],
                libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
            )[0],
        )
        for s in libraries["All"]["fc_gene_avg_bin"]
    ]
)


# Plot comulative dependencies
#

plot_df = pd.concat([libraries[c]["cumulative_dep"].assign(dtype=c) for c in libraries])
plot_df = plot_df.groupby(["dtype", "n_samples"])["n_deps"].mean().reset_index()

plt.figure(figsize=(2, 2))

for i, (dtype, df) in enumerate(plot_df.groupby("dtype")):
    plt.plot(df["n_samples"], df["n_deps"], label=dtype, color=CrispyPlot.PAL_DBGD[i])

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Number cell lines")
plt.ylabel(f"Number dependencies\n{(FDR_THRES*100):.0f}% FPR")

plt.legend(frameon=False)
plt.savefig(f"{RPATH}/MinimalLib_benchmark_cumsum.pdf", bbox_inches="tight")
plt.close("all")


# Gene fold-change correlation
#

plt.figure(figsize=(2.5, 1.5), dpi=600)
sns.distplot(
    deprecovered["spearman"],
    hist=False,
    kde_kws={"cut": 0, "shade": True},
    color="k",
)
plt.xlabel("Spearman's R")
plt.title("Samples gene fold-change correlation")
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.savefig(
    f"{RPATH}/MinimalLib_benchmark_dependencies_fc_corr_histogram.pdf", bbox_inches="tight"
)
plt.close("all")


# Effect sizes of recap dependencies
#

plot_df = pd.concat([
    libraries["All"]["fc_gene_avg_bin"].loc[genes].unstack().rename("All_bin"),
    libraries["Minimal"]["fc_gene_avg_bin"].loc[genes].unstack().rename("Minimal_bin"),
    libraries["All"]["fc_gene_avg_scl"].loc[genes].unstack().rename("All_fc"),
], axis=1, sort=False).reset_index()
plot_df["Recap"] = [a if a != m else sum([a, m]) for a, m in plot_df[["All_bin", "Minimal_bin"]].values]
plot_df["Recap"] = plot_df["Recap"].replace({2: "Yes", 1: "No", 0: "NA"})

order = ["Yes", "No"]
pal = {"Yes": CrispyPlot.PAL_DBGD[1], "No": CrispyPlot.PAL_DBGD[0]}
samples = set(plot_df["model_id"])

sns.boxplot(
    "Recap",
    "All_fc",
    data=plot_df,
    order=order,
    palette=pal,
    saturation=1,
    showcaps=False,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    flierprops=dict(
        marker="o",
        markerfacecolor="black",
        markersize=1.0,
        linestyle="none",
        markeredgecolor="none",
        alpha=0.6,
    ),
    notch=True,
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")
plt.xlabel("Dependency recapitulated")
plt.ylabel("All sgRNAs fold-change (scaled)")
plt.gcf().set_size_inches(.75, 2.)
plt.savefig(
    f"{RPATH}/MinimalLib_benchmark_dependencies_effect_sizes_boxplots.pdf", bbox_inches="tight"
)
plt.close("all")


# AROC and FC threshold scatters
#

for metric in ["auc", "thres"]:
    plot_df = pd.concat(
        [libraries[l]["fpr"][metric].rename(l) for l in libraries], axis=1
    )
    plot_df["density"] = density_interpolate(plot_df["All"], plot_df["Minimal"])
    plot_df = plot_df.sort_values("density")

    x_min, x_max = plot_df.iloc[:, :2].min().min(), plot_df.iloc[:, :2].max().max()

    plt.scatter(
        plot_df["All"],
        plot_df["Minimal"],
        c=plot_df["density"],
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=3,
        alpha=0.7,
    )

    lims = [x_max, x_min]
    plt.plot(lims, lims, "k-", lw=0.3, zorder=0)

    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    plt.xlabel("All sgRNAs")
    plt.ylabel("Minimal Library - 2 sgRNAs")
    plt.title(
        f"AROC essential genes {(FDR_THRES*100):.0f}% FDR"
        if metric == "auc"
        else f"Fold-change threshold {(FDR_THRES*100):.0f}% FDR"
    )

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(
        f"{RPATH}/MinimalLib_benchmark_scatter_{metric}.pdf", bbox_inches="tight"
    )
    plt.close("all")


# Number of dependencies scatter
#

plot_df = pd.concat(
    [libraries[l]["fc_gene_avg_bin"].sum(1).rename(l) for l in libraries],
    axis=1,
    sort=False,
).dropna()
plot_df.to_csv(f"{RPATH}/MinimalLib_benchmark_dependencies_per_gene.csv")
plot_df["density"] = density_interpolate(plot_df["All"], plot_df["Minimal"])
plot_df = plot_df.sort_values("density")

x_min, x_max = plot_df.iloc[:, :2].min().min(), plot_df.iloc[:, :2].max().max()

ax = plt.gca()

ax.scatter(
    plot_df["All"],
    plot_df["Minimal"],
    c=plot_df["density"],
    marker="o",
    edgecolor="",
    cmap="Spectral_r",
    s=3,
    alpha=0.7,
)

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

lims = [x_max, x_min]
ax.plot(lims, lims, "k-", lw=0.3, zorder=0)

cor, pval = pearsonr(plot_df["All"], plot_df["Minimal"])
annot_text = f"Spearman's R={cor:.2g}"

ax.text(0.05, 0.95, annot_text, fontsize=4, transform=ax.transAxes, ha="left")

ax.set_xlabel("All sgRNAs")
ax.set_ylabel("Minimal Library - 2 sgRNAs")
ax.set_title("Number of dependencies per gene")

plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{RPATH}/MinimalLib_benchmark_dependencies_recap_sctter.pdf", bbox_inches="tight"
)
plt.close("all")


# Recover plot
#

fig, ax1 = plt.subplots(figsize=(2.5, 2))

plot_df = deprecovered.sort_values("recall", ascending=False).reset_index(drop=True)

ax1.scatter(plot_df.index, plot_df["recall"], c="k", s=3)

for t in [0.8, 0.9]:
    s_t = plot_df[plot_df["recall"] < t].index[0]
    ax1.axvline(s_t, lw=0.1, c="k", zorder=0)

    s_t_text = f"{((s_t / plot_df.shape[0]) * 100):.1f}%"
    ax1.text(s_t, plt.ylim()[0], s_t_text, fontsize=6, c="k")

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.xlabel("Number of cell lines")
plt.ylabel(f"Recall dependencies\n{(FDR_THRES*100):.0f}% FDR")

plt.savefig(f"{RPATH}/MinimalLib_benchmark_recall_plot.pdf", bbox_inches="tight")
plt.close("all")


# Discrepancy examples between Minimal Library and All guides
#

x = pd.concat(
    [libraries[c]["fc_gene_avg_bin"].loc["WRN"].rename(c) for c in libraries], axis=1
)
x = pd.concat([x, Sample().samplesheet.loc[x.index, "msi_status"]], axis=1).dropna()
x.sort_values(["All", "Minimal", "msi_status"], ascending=[False, False, True])

gene_deps = pd.read_csv(f"{RPATH}/MinimalLib_benchmark_dependencies_per_gene.csv", index_col=0)
gene_deps = gene_deps.loc[gene_deps.eval("All - Minimal").sort_values().index]

g = "RPL9"

g_metrics = ky_lib[ky_lib["Approved_Symbol"] == g].sort_values("KS")
g_metrics["MinimalLib"] = g_metrics.index.isin(ml_lib.index).astype(int)

g_fcs = libraries["All"]["fc"].loc[g_metrics.index]

sns.clustermap(
    g_fcs,
    cmap="Spectral",
    xticklabels=False,
    figsize=(2, 2),
    row_colors=[CrispyPlot.PAL_DBGD[i] for i in g_metrics["MinimalLib"]],
    center=0,
)

plt.savefig(f"{RPATH}/MinimalLib_benchmark_mismatch_example_{g}.pdf", bbox_inches="tight")
plt.close("all")
