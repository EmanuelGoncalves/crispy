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
from math import sqrt
from crispy.QCPlot import QCplot
from crispy.Utils import Utils
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error
from crispy.CopyNumberCorrection import Crispy
from scipy.stats import spearmanr, mannwhitneyu
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from crispy.DataImporter import Mobem, CopyNumber, CopyNumberSegmentation
from C50K import (
    clib_palette,
    clib_order,
    define_sgrnas_sets,
    estimate_ks,
    sgrnas_scores_scatter,
    guides_aroc_benchmark,
    ky_v11_calculate_gene_fc,
    lm_associations,
    project_score_sample_map,
    assemble_crisprcleanr,
    assemble_crispy,
)


# -
dpath = pkg_resources.resource_filename("crispy", "data/")
rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")


# Gene sets fold-change distributions

genesets = {
    "essential": Utils.get_essential_genes(return_series=False),
    "nonessential": Utils.get_non_essential_genes(return_series=False),
}

genesets = {
    m: {g: metrics_fc[m].reindex(genesets[g]).median(1).dropna() for g in genesets}
    for m in metrics_fc
}


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

palette = dict(Yes="#d62728", No="#E1E1E1")

f, axs = plt.subplots(
    1,
    len(guides_opt_corr),
    sharex="all",
    sharey="all",
    figsize=(len(guides_opt_corr) * 0.7, 1.5),
    dpi=600,
)

all_rep_corr = guides_opt_corr["all"].query("replicate == 'Yes'")["corr"].median()

for i, (n, df) in enumerate(guides_opt_corr.items()):
    ax = axs[i]

    sns.boxplot(
        "replicate",
        "corr",
        data=df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
        notch=True,
        ax=ax,
    )

    ax.axhline(all_rep_corr, ls="-", lw=1.0, zorder=0, c="#484848")

    ax.set_title(n)
    ax.set_xlabel("Replicate")
    ax.set_ylabel("Pearson R" if i == 0 else None)

plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/ky_v11_rep_corrs.png", bbox_inches="tight")
plt.close("all")


# ANOVA scatter plots

metrics = list(metrics_fc)[:-1]

f, axs = plt.subplots(
    1,
    len(metrics),
    sharex="all",
    sharey="all",
    figsize=(len(metrics) * 1.5, 1.5),
    dpi=600,
)

x_min, x_max = lm_assocs["beta"].min(), lm_assocs["beta"].max()

y_var = lm_assocs.query(f"metric == 'all'").set_index(["gene", "phenotype"])

for i, m in enumerate(metrics):
    ax = axs[i]

    x_var_m = lm_assocs.query(f"metric == '{m}'").set_index(["gene", "phenotype"])

    var_fdr = pd.concat([x_var_m["fdr"].rename("x"), y_var["fdr"].rename("y")], axis=1)
    var_fdr["signif"] = (var_fdr < 0.1).sum(1)
    var_fdr["size"] = (var_fdr["signif"] + 1) * 3
    var_fdr["color"] = var_fdr["signif"].replace(
        {0: "#e1e1e1", 1: "#ffbb78", 2: "#ff7f0e"}
    )
    var_fdr["marker"] = [
        "o" if s != 1 else ("v" if x < 0.1 else "<")
        for x, y, s in var_fdr[["x", "y", "signif"]].values
    ]

    data, x_e, y_e = np.histogram2d(
        x_var_m["beta"], y_var.loc[x_var_m.index, "beta"], bins=20
    )
    z_var = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x_var_m["beta"], y_var.loc[x_var_m.index, "beta"]]).T,
        method="splinef2d",
        bounds_error=False,
    )

    ax.scatter(
        x_var_m["beta"],
        y_var.loc[x_var_m.index, "beta"],
        c=z_var,
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=var_fdr.loc[x_var_m.index, "size"],
        alpha=0.3,
    )

    for n, df in var_fdr.groupby("signif"):
        if n == 1:
            for marker, df_m in df.groupby("marker"):
                ax.scatter(
                    x_var_m.loc[df_m.index, "beta"],
                    y_var.loc[df_m.index, "beta"],
                    c=df_m.iloc[0]["color"],
                    marker=marker,
                    edgecolor="white",
                    lw=0.3,
                    s=df_m.iloc[0]["size"],
                    alpha=1,
                    zorder=2,
                    label="Partial" if n == 1 else "Both",
                )

    rmse = sqrt(mean_squared_error(x_var_m["beta"], y_var.loc[x_var_m.index, "beta"]))
    cor, _ = spearmanr(x_var_m["beta"], y_var.loc[x_var_m.index, "beta"])
    annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"

    ax.text(0.95, 0.05, annot_text, fontsize=4, transform=axs[i].transAxes, ha="right")

    lims = [x_max, x_min]
    ax.plot(lims, lims, "k-", lw=0.3, zorder=0)
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    ax.set_title(m)
    ax.set_xlabel("Effect size (beta)")
    ax.set_ylabel("All Guides\nEffect size (beta)" if i == 0 else None)

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 4})

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_anova.png", bbox_inches="tight")
plt.close("all")


# ANOVA top missed/found biormarker associations

plot_df = pd.concat(
    [
        lm_assocs.query(f"metric == '{m}'")
        .set_index(["gene", "phenotype"])
        .add_prefix(f"{m}_")
        for m in ["all", "ks"]
    ],
    axis=1,
    sort=False,
)
plot_df = plot_df[(plot_df[["all_fdr", "ks_fdr"]] < 0.1).sum(1) == 1]
plot_df["delta_beta_abs"] = plot_df.eval("all_beta - ks_beta").abs()

n_top = 10

plot_df = [
    (
        "Top missed associations",
        "top_missed",
        plot_df.query("all_fdr < 0.1")
        .sort_values("delta_beta_abs", ascending=False)
        .head(n_top),
    ),
    (
        "Top found associations",
        "top_found",
        plot_df.query("ks_fdr < 0.1")
        .sort_values("delta_beta_abs", ascending=False)
        .head(n_top),
    ),
]

for (assoc_name, assoc_file, assoc_top) in plot_df:
    n_cols = 5
    n_rows = int(np.ceil(n_top / n_cols))

    f, axs = plt.subplots(
        n_rows,
        n_cols,
        sharex="all",
        sharey="all",
        figsize=(n_cols * 1.5, n_rows * 1.5),
        dpi=600,
    )

    for i, ((feature, phenotype), row) in enumerate(assoc_top.iterrows()):
        df = pd.concat(
            [
                x_mobem.loc[samples, feature].rename("feature"),
                metrics_fc["ks"].loc[phenotype, samples].rename("ks"),
                metrics_fc["all"].loc[phenotype, samples].rename("all"),
            ],
            axis=1,
            sort=False,
        )
        df = pd.melt(df, id_vars="feature", value_vars=["ks", "all"])

        row_i = int(np.floor(i / n_cols))
        col_i = i % n_cols
        ax = axs[row_i][col_i]

        sns.boxplot(
            "variable",
            "value",
            "feature",
            data=df,
            order=["all", "ks"],
            palette={1: "#d62728", 0: "#E1E1E1"},
            saturation=1,
            showcaps=False,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            "variable",
            "value",
            "feature",
            data=df,
            order=["all", "ks"],
            split=True,
            edgecolor="white",
            linewidth=0.1,
            size=2,
            palette={1: "#d62728", 0: "#E1E1E1"},
            ax=ax,
        )

        for g in sgrna_sets:
            ax.axhline(
                sgrna_sets[g]["fc"].median(),
                ls="-",
                lw=0.3,
                c=sgrna_sets[g]["color"],
                zorder=0,
            )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
        ax.set_xlabel("Guide selection" if row_i == (n_rows - 1) else None)
        ax.set_ylabel(f"Fold-change" if col_i == 0 else None)
        ax.set_title(f"{feature} ~ {phenotype}")
        ax.get_legend().remove()

        all_annot_text = f"beta={row['all_beta']:.1f}\nFDR={row['all_fdr']:.2g}"
        ax.text(
            0.3, 0.1, all_annot_text, fontsize=4, transform=ax.transAxes, ha="center"
        )

        ks_annot_text = f"beta={row['ks_beta']:.1f}\nFDR={row['ks_fdr']:.2g}"
        ax.text(
            0.8, 0.1, ks_annot_text, fontsize=4, transform=ax.transAxes, ha="center"
        )

    plt.suptitle(assoc_name, y=1.05)
    plt.subplots_adjust(hspace=0.25, wspace=0.05)
    plt.savefig(f"{rpath}/ky_v11_anova_examples_{assoc_file}.png", bbox_inches="tight")
    plt.close("all")


# Gene sets distributions

f, axs = plt.subplots(
    1,
    len(metrics_fc),
    sharex="all",
    sharey="all",
    figsize=(len(metrics_fc) * 1.5, 1.5),
    dpi=600,
)

for i, m in enumerate(genesets):
    ax = axs[i]

    s, p = mannwhitneyu(genesets[m]["essential"], genesets[m]["nonessential"])

    for n in genesets[m]:
        sns.distplot(
            genesets[m][n],
            hist=False,
            label=n,
            kde_kws={"cut": 0, "shade": True},
            color=sgrna_sets[n]["color"],
            ax=ax,
        )
        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
        ax.set_title(f"{m}")
        ax.set_xlabel("sgRNAs fold-change")
        ax.text(
            0.05,
            0.75,
            f"Mann-Whitney p-val={p:.2g}",
            fontsize=4,
            transform=ax.transAxes,
            ha="left",
        )
        ax.legend(frameon=False, prop={"size": 4}, loc=2)

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_metrics_gene_distributions.png", bbox_inches="tight")
plt.close("all")

