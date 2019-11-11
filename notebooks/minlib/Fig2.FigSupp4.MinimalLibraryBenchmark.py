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

import ast
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.QCPlot import QCplot
from crispy.DataImporter import Sample
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr, pearsonr, ttest_ind
from crispy.CRISPRData import CRISPRDataSet, Library, ReadCounts
from minlib.Utils import project_score_sample_map, density_interpolate
from sklearn.metrics import (
    recall_score,
    precision_score,
    average_precision_score,
    precision_recall_curve,
)


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

NGUIDES, REMOVE_DISCORDANT = 2, True
ml_lib_name = (
    f"MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz"
)
ml_lib = Library.load_library(ml_lib_name).query("Library == 'KosukeYusa'")
ml_lib = ml_lib[ml_lib.index.isin(ky_counts.index)]
ml_lib = ml_lib.loc[[i for i in ml_lib.index if not i.startswith("CTRL0")]]


# Genes overlap
#

genes = set(ky_lib["Approved_Symbol"]).intersection(ml_lib["Approved_Symbol"])
LOG.info(f"Genes={len(genes)}")


# Comparisons
#

FDR_THRES = 0.01

libraries = dict(
    All=dict(name="All", lib=ky_lib), Minimal=dict(name="Minimal", lib=ml_lib)
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
    fpr["auc"] = [
        QCplot.aroc_threshold(fc_gene_avg[s], fpr_thres=0.2)[0] for s in fpr.index
    ]

    # Gene binarised
    fc_gene_avg_bin = (fc_gene_avg[fpr.index] < fpr["thres"]).astype(int)

    # Store
    libraries[ltype]["fc"] = fc
    libraries[ltype]["fc_gene"] = fc_gene
    libraries[ltype]["fc_gene_avg"] = fc_gene_avg
    libraries[ltype]["fc_gene_avg_scl"] = fc_gene_avg_scl
    libraries[ltype]["fc_gene_avg_bin"] = fc_gene_avg_bin
    libraries[ltype]["fpr"] = fpr


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
            ap=average_precision_score(
                libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
                -libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
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

sample_qc = pd.read_excel(
    f"{DPATH}/crispr_manifests/project_score_sample_qc.xlsx", index_col=0
)
sample_qc["Transduction Efficiencty"] = sample_qc["Transduction Efficiencty"].apply(
    lambda v: np.nanmean(
        [np.nan if i == "NA" else float(i) for i in str(v).split(", ")]
    )
)
sample_qc["Cas9 Activity"] = sample_qc["Cas9 Activity"].apply(
    lambda v: np.nanmean(
        [np.nan if i == "NA" else float(i) for i in str(v).split(", ")]
    )
)

deprecovered = (
    pd.concat(
        [
            deprecovered.set_index("samples"),
            sample_qc["Transduction Efficiencty"],
            sample_qc["Cas9 Activity"],
            sample_qc["CellLine Passing LowLev QC"],
            Sample().samplesheet["tissue"],
        ],
        axis=1,
        sort=False,
    )
    .dropna(subset=["ap"])
    .sort_values("ap", ascending=False)
)
deprecovered["Efficacy"] = (
    (deprecovered["Transduction Efficiencty"] / 100)
    * (deprecovered["Cas9 Activity"] / 100)
    * 100
)
deprecovered.to_excel(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_qc_sample_metrics.xlsx"
)


# Disconcordance between number of dependent cell lines found per gene
#

gene_deps = pd.concat(
    [libraries[l]["fc_gene_avg_bin"].sum(1).rename(l) for l in libraries],
    axis=1,
    sort=False,
).dropna()
gene_deps["difference"] = gene_deps.eval("All - Minimal")
gene_deps = gene_deps.loc[gene_deps.sort_values("difference").index]
gene_deps["match"] = (gene_deps["difference"].abs() > 100).replace(
    {True: "Disconcordant", False: "Concordant"}
)
gene_deps["n_guides"] = (
    ky_lib["Approved_Symbol"].dropna().value_counts().loc[gene_deps.index].values
)
gene_deps.to_excel(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_gene_concordance.xlsx"
)


# Sample QC scatter plots
#

y_var = "ap"
x_vars = [
    "Transduction Efficiencty",
    "Cas9 Activity",
    "Replicates correlation",
    "Efficacy",
]

f, axs = plt.subplots(
    1, len(x_vars), sharex="none", sharey="row", figsize=(len(x_vars) * 2, 2)
)

for i, x_var in enumerate(x_vars):
    ax = axs[i]

    plot_df = pd.concat(
        [deprecovered[x_var], deprecovered[y_var]], axis=1, sort=False
    ).dropna()
    plot_df["density"] = density_interpolate(plot_df[x_var], plot_df[y_var])
    plot_df = plot_df.sort_values("density")

    ax.scatter(
        plot_df[x_var],
        plot_df[y_var],
        c=plot_df["density"],
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=3,
        alpha=0.7,
    )

    sns.regplot(
        plot_df[x_var],
        plot_df[y_var],
        scatter=False,
        color=CrispyPlot.PAL_DBGD[0],
        truncate=True,
        ax=ax,
    )

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    ax.set_ylabel("Average precision score" if i == 0 else None)

    cor, pval = spearmanr(plot_df[x_var], plot_df[y_var])
    annot_text = f"Spearman's R={cor:.2g}, p-val={pval:.1g}"
    ax.text(0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_qc_scatters.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Gene fold-change correlation
#

plt.figure(figsize=(2.5, 1.5), dpi=600)
sns.distplot(
    deprecovered["spearman"], hist=False, kde_kws={"cut": 0, "shade": True}, color="k"
)
plt.xlabel("Spearman's R")
plt.title(
    f"Samples gene fold-change correlation\n(mean={deprecovered['spearman'].mean():.2f})"
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_dependencies_fc_corr_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Effect sizes of recap dependencies
#

plot_df = pd.concat(
    [
        libraries["All"]["fc_gene_avg_bin"].loc[genes].unstack().rename("All_bin"),
        libraries["Minimal"]["fc_gene_avg_bin"]
        .loc[genes]
        .unstack()
        .rename("Minimal_bin"),
        libraries["All"]["fc_gene_avg_scl"].loc[genes].unstack().rename("All_fc"),
    ],
    axis=1,
    sort=False,
).reset_index()
plot_df["Recap"] = [
    a if a != m else sum([a, m]) for a, m in plot_df[["All_bin", "Minimal_bin"]].values
]
plot_df["Recap"] = plot_df["Recap"].replace({2: "Yes", 1: "No", 0: "NA"})

ttest_ind(
    plot_df.query("Recap == 'Yes'")["All_fc"],
    plot_df.query("Recap == 'No'")["All_fc"],
    equal_var=False,
)

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
plt.gcf().set_size_inches(0.75, 2.0)
plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_dependencies_effect_sizes_boxplots.pdf",
    bbox_inches="tight",
    transparent=True,
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
    plt.title(f"AROC essential genes" if metric == "auc" else f"Fold-change threshold")

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(
        f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_scatter_{metric}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


# Number of dependencies scatter
#

gene_deps["density"] = density_interpolate(gene_deps["All"], gene_deps["Minimal"])
gene_deps["n_guides"] = (
    ky_lib["Approved_Symbol"].dropna().value_counts().loc[gene_deps.index].values
)
gene_deps = gene_deps.sort_values("density")

x_min, x_max = gene_deps.iloc[:, :2].min().min(), gene_deps.iloc[:, :2].max().max()

ax = plt.gca()

for t, df in gene_deps.groupby("match"):
    ax.scatter(
        df["All"],
        df["Minimal"],
        c=df["density"],
        marker="o",
        edgecolor="#e34a33" if t == "Disconcordant" else "",
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
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_dependencies_recap_sctter.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Recover plot
#

pal = pd.read_excel(f"{DPATH}/tissue_palette.xlsx", index_col=0)["Color"]

plot_df = deprecovered.reset_index(drop=True)

_, ax1 = plt.subplots(figsize=(3, 2))

for t, df in plot_df.groupby("tissue"):
    ax1.scatter(df.index, df["ap"], c=pal[t], s=5, label=t)

for t in [0.8, 0.9]:
    s_t = plot_df[plot_df["ap"] < t].index[0]
    ax1.axvline(s_t, lw=0.1, c="k", zorder=0)

    s_t_text = f"{((s_t / plot_df.shape[0]) * 100):.1f}%"
    ax1.text(s_t, plt.ylim()[0], s_t_text, fontsize=6, c="k")

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.xlabel("Number of cell lines")
plt.ylabel(f"Average precission score")

plt.legend(frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5))

plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_recall_plot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Recall-Precision curves examples
#

_, ax = plt.subplots(figsize=(2, 2))

for s in deprecovered.index:
    c_precision, c_recall, _ = precision_recall_curve(
        libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
        -libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
    )
    ax.plot(c_recall, c_precision, color=CrispyPlot.PAL_DBGD[2], alpha=0.4)

for i, s in enumerate([deprecovered.tail(1).index[0], deprecovered.head(1).index[0]]):
    c_precision, c_recall, _ = precision_recall_curve(
        libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
        -libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
    )

    ap = average_precision_score(
        libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
        -libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
    )

    ax.plot(
        c_recall, c_precision, label=f"{s} (AP={ap:.2f})", color=CrispyPlot.PAL_DBGD[i]
    )

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.legend(prop={"size": 4}, frameon=False, loc=3)

plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_recall_precision_curve_examples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Discrepancy examples between Minimal Library and All guides
#

# g, g_diff = "VHL", -82
for g, g_diff in (
    gene_deps.sort_values("difference")
    .iloc[np.r_[0:5, -5:0]]
    .reset_index()[["index", "difference"]]
    .values
):

    g_metrics = ky_lib[ky_lib["Approved_Symbol"] == g].sort_values("KS")
    g_metrics["MinimalLib"] = g_metrics.index.isin(ml_lib.index).astype(int)
    g_metrics["Off_Target"] = g_metrics["Off_Target"].apply(ast.literal_eval).values
    g_metrics["OT_0"] = [g_metrics.loc[i, "Off_Target"][0] for i in g_metrics.index]
    g_metrics["OT_1"] = [g_metrics.loc[i, "Off_Target"][1] for i in g_metrics.index]

    g_fcs = libraries["All"]["fc"].loc[g_metrics.index]

    gx = sns.clustermap(g_fcs, cmap="Spectral", xticklabels=False, figsize=(2, 2))
    plt.close("all")

    metrics = ["MinimalLib", "JACKS", "RuleSet2", "KS", "OT_0", "OT_1"]

    f, axs = plt.subplots(
        1,
        len(metrics) + 1,
        sharex="none",
        sharey="none",
        figsize=(2 + 0.5 * len(metrics), 2),
        gridspec_kw={"width_ratios": [3] + [1] * len(metrics)},
    )

    for i, ax in enumerate(axs):
        if i == 0:
            sns.heatmap(
                gx.data2d,
                xticklabels=False,
                cmap="Spectral",
                center=0,
                cbar=False,
                ax=ax,
            )

        else:
            metric = metrics[i - 1]
            center = 1 if metric == "JACKS" else None
            cmap = "RdBu_r" if metric == "JACKS" else "GnBu"
            fmt = ".1f" if metric not in ["MinimalLib", "OT_0", "OT_1"] else ".0f"

            sns.heatmap(
                g_metrics.loc[gx.data2d.index, [metric]],
                xticklabels=False,
                cmap=cmap,
                center=center,
                cbar=False,
                annot=True,
                fmt=fmt,
                ax=ax,
            )

            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_ylabel(None)
        ax.set_title("Fold-change" if i == 0 else metric)

    plt.suptitle(
        f"{g}\nDifference between number of dependencies\n(All - Minimal = {g_diff})",
        y=1.2,
    )

    plt.savefig(
        f"{RPATH}/{ml_lib_name.split('.')[0]}_benchmark_mismatch_example_{g}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
