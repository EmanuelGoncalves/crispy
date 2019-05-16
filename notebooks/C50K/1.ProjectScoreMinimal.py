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

import datetime
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from C50K import (
    clib_palette,
    clib_order,
    define_sgrnas_sets,
    estimate_ks,
    sgrnas_scores_scatter,
    guides_aroc_benchmark,
)


# -
dpath = pkg_resources.resource_filename("crispy", "data/")
rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

# CRISPR libraries size

clib_size = pd.read_excel(f"{rpath}/Human CRISPR-Cas9 dropout libraries.xlsx")


# CRISPR Project Score - Kosuke Yusa v1.1

ky_v11_data = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_data.counts.remove_low_counts(ky_v11_data.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)


# Sets of sgRNAs

sgrna_sets = define_sgrnas_sets(ky_v11_data.lib, ky_v11_fc, add_controls=True)


# Kolmogorov-Smirnov statistic guide comparison

ky_v11_ks = estimate_ks(ky_v11_fc, sgrna_sets["nontargeting"]["fc"])


# sgRNAs Efficiency scores

jacks = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0.csv", index_col=0)
doenchroot = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0_DoenchRoot.csv", index_col=0)
forecast = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0_ForecastFrameshift.csv", index_col=0)

ky_v11_ks["jacks"] = jacks.reindex(ky_v11_ks.index)["X1"].values
ky_v11_ks["doenchroot"] = doenchroot.reindex(ky_v11_ks.index)["score_with_ppi"].values
ky_v11_ks["forecast"] = forecast.reindex(ky_v11_ks.index)["In Frame Percentage"].values
ky_v11_ks["Gene"] = ky_v11_data.lib.reindex(ky_v11_ks.index)["Gene"].values


# Benchmark

nguides_thres = 5

ky_v11_metrics = ky_v11_ks.dropna()

jacks_metric = abs(ky_v11_metrics["jacks"] - 1)

ks_metric = 1 - ky_v11_metrics["ks_control"]
ks_metric_inverted = ky_v11_metrics["ks_control"]

fc_metric = ky_v11_metrics["median_fc"] - ky_v11_metrics["median_fc"].min()

ky_v11_metrics = pd.DataFrame(
    dict(
        jacks=jacks_metric,
        doenchroot=ky_v11_metrics["doenchroot"],
        ks=ks_metric,
        ks_inverse=ks_metric_inverted,
        median_fc=fc_metric,
        Gene=ky_v11_metrics["Gene"],
    )
)

ky_v11_arocs = guides_aroc_benchmark(
    ky_v11_metrics,
    ky_v11_fc,
    nguides_thres=nguides_thres,
    guide_set=set(ky_v11_metrics.index),
    rand_iterations=10
)

ky_v11_arocs.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_aroc.xlsx", index=False)


# - Plot
# CRISPR Library sizes

plt.figure(figsize=(2.5, 2), dpi=600)
sns.scatterplot(
    "Date",
    "Number of guides",
    "Library ID",
    data=clib_size,
    size="sgRNAs per Gene",
    palette=clib_palette,
    hue_order=clib_order,
    sizes=(10, 50),
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
plt.xlim(datetime.date(2013, 11, 1), datetime.date(2020, 3, 1))
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 4})
plt.title("CRISPR-Cas9 libraries")
plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/crispr_libraries_coverage.pdf", bbox_inches="tight")
plt.close("all")


# Guide sets distributions

plt.figure(figsize=(2.5, 2), dpi=600)
for c in sgrna_sets:
    sns.distplot(
        sgrna_sets[c]["fc"],
        hist=False,
        label=c,
        kde_kws={"cut": 0, "shade": True},
        color=sgrna_sets[c]["color"],
    )
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("sgRNAs fold-change")
plt.legend(frameon=False)
plt.title("Project Score - KY v1.1")
plt.savefig(f"{rpath}/ky_v11_guides_distributions.pdf", bbox_inches="tight")
plt.close("all")


# K-S metric

sgrnas_scores_scatter(ky_v11_ks)

for gset in sgrna_sets:
    plt.axhline(
        sgrna_sets[gset]["fc"].median(),
        ls="-",
        lw=1,
        c=sgrna_sets[gset]["color"],
        zorder=0,
    )

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
plt.xlabel("sgRNA Kolmogorov-Smirnov (non-targeting)")
plt.ylabel("sgRNA median Fold-change")
plt.title("Project Score - KY v1.1")
plt.savefig(f"{rpath}/ky_v11_ks_scatter_median.png", bbox_inches="tight")
plt.close("all")


# K-S metric guides distribution

f, axs = plt.subplots(1, 3, sharex="all", sharey="all", figsize=(4.5, 1.5), dpi=600)

for i, gset in enumerate(sgrna_sets):
    ax = axs[i]
    sgrnas_scores_scatter(
        ky_v11_ks,
        z=sgrna_sets[gset]["sgrnas"],
        z_color=sgrna_sets[gset]["color"],
        ax=ax,
    )
    for g in sgrna_sets:
        ax.axhline(
            sgrna_sets[g]["fc"].median(),
            ls="-",
            lw=1,
            c=sgrna_sets[g]["color"],
            zorder=0,
        )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_title(gset)
    ax.set_xlabel("sgRNA KS\n(non-targeting)")
    ax.set_ylabel("sgRNAs fold-change\n(median)" if i == 0 else None)

plt.suptitle("Project Score - KY v1.1", y=1.07)
plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/ky_v11_ks_scatter_median_guides.png", bbox_inches="tight")
plt.close("all")


# Metrics correlation

f, axs = plt.subplots(1, 4, sharex="none", sharey="all", figsize=(6.0, 1.5), dpi=600)

for i, (x, y, n) in enumerate(
    [
        ("ks_control", "median_fc", "K-S statistic"),
        ("jacks", "median_fc", "JACKS scores"),
        ("doenchroot", "median_fc", "Doench-Root score"),
        ("forecast", "median_fc", "FORECAST\nIn Frame Percentage"),
    ]
):
    ax = axs[i]

    sgrnas_scores_scatter(ky_v11_ks, x, y, add_cbar=False, ax=ax)

    for g in sgrna_sets:
        ax.axhline(
            sgrna_sets[g]["fc"].median(),
            ls="-",
            lw=1,
            c=sgrna_sets[g]["color"],
            zorder=0,
        )

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_xlabel(n)
    ax.set_ylabel(None if i != 0 else "sgRNAs fold-change\n(median)")

plt.suptitle("Project Score - KY v1.1", y=1.07)
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_ks_scatter_metrics.png", bbox_inches="tight")
plt.close("all")


# Benchmark AROC

x_feature = "ks"
y_features = ["jacks", "doenchroot", "median_fc", "ks_inverse", "random"]

f, axs = plt.subplots(
    len(y_features),
    nguides_thres,
    sharex="all",
    sharey="all",
    figsize=(6, len(y_features)),
    dpi=600,
)

x_min, x_max = ky_v11_arocs["auc"].min(), ky_v11_arocs["auc"].max()

for i in range(nguides_thres):
    plot_df = ky_v11_arocs.query(f"n_guides == {(i + 1)}")

    x_var = (
        plot_df.query(f"dtype == '{x_feature}'").set_index("index")["auc"].rename("x")
    )
    y_vars = pd.concat(
        [
            plot_df.query(f"dtype == '{f}'")
            .set_index("index")["auc"]
            .rename(f"y{i + 1}")
            for i, f in enumerate(y_features)
        ],
        axis=1,
    )
    plot_df = pd.concat([x_var, y_vars], axis=1)

    for j in range(len(y_features)):
        data, x_e, y_e = np.histogram2d(plot_df["x"], plot_df[f"y{j + 1}"], bins=20)
        z_var = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([plot_df["x"], plot_df[f"y{j + 1}"]]).T,
            method="splinef2d",
            bounds_error=False,
        )

        axs[j][i].scatter(
            plot_df["x"],
            plot_df[f"y{j + 1}"],
            c=z_var,
            marker="o",
            edgecolor="",
            cmap="Spectral_r",
            s=3,
            alpha=0.7,
        )

        lims = [x_max, x_min]
        axs[j][i].plot(lims, lims, "k-", lw=0.3, zorder=0)

    axs[0][i].set_title(f"#(guides) = {i + 1}")

    if i == 0:
        for l_index, l in enumerate(y_features):
            axs[l_index][i].set_ylabel(f"{l}\n(AROC)")

    axs[len(y_features) - 1][i].set_xlabel("KS (AROC)")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_ks_guides_benchmark_scatter.png", bbox_inches="tight")
plt.close("all")


# Replicates correlation

def replicates_correlation(fc_sgrna, guides=None):
    if guides is not None:
        df = fc_sgrna.reindex(guides)
    else:
        df = fc_sgrna.copy()

    df.columns = [c.split("_")[0] for c in df]

    df_corr = df.corr()
    df_corr = df_corr.where(np.triu(np.ones(df_corr.shape), 1).astype(np.bool))
    df_corr = df_corr.unstack().dropna().reset_index()
    df_corr.columns = ["sample_1", "sample_2", "corr"]

    df_corr["replicate"] = (
        (df_corr["sample_1"] == df_corr["sample_2"])
        .replace({True: "Yes", False: "No"})
        .values
    )

    return df_corr


guides_all = ky_v11_ks.dropna(subset=["ks_control"])

guides_best = guides_all.sort_values("ks_control", ascending=False).groupby("Gene").head(n=2)
guides_worst = guides_all.sort_values("ks_control", ascending=True).groupby("Gene").head(n=2)

corr_all = replicates_correlation(ky_v11_fc, guides=list(guides_all.index))
corr_best = replicates_correlation(ky_v11_fc, guides=list(guides_best.index))
corr_worst = replicates_correlation(ky_v11_fc, guides=list(guides_worst.index))

#
palette = dict(Yes="#d62728", No="#E1E1E1")

f, axs = plt.subplots(3, 1, sharex="all", sharey="all", figsize=(2, 2), dpi=600)

for i, (n, df) in enumerate([("all", corr_all), ("best n=2", corr_best), ("worst n=2", corr_worst)]):
    ax = axs[i]

    sns.boxplot(
        "corr",
        "replicate",
        orient="h",
        data=df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
        notch=True,
        ax=ax,
    )
    ax.axvline(0, ls="-", lw=0.05, zorder=0, c="#484848")
    ax.set_ylabel("Replicate")

    ax_twin = ax.twinx()
    ax_twin.set_ylabel(n.capitalize())
    ax_twin.axes.get_yaxis().set_ticks([])

ax.set_xlabel("Pearson R")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_rep_corrs.pdf", bbox_inches="tight")
plt.close("all")
