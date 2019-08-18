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
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy import CrispyPlot, QCplot, Utils
from crispy.CRISPRData import CRISPRDataSet, ReadCounts


if __name__ == "__main__":
    organoids = dict(
        name="Combined set of organoids",
        read_counts="bme2_counts.csv.gz",
        library="Yusa_v1.1.csv.gz",
        plasmids=["Plasmid_v1.1"],
        samplesheet="samplesheet.xlsx",
    )

    # - Imports
    ddir = pkg_resources.resource_filename("data", "organoids/bme2/")
    dreports = pkg_resources.resource_filename("notebooks", "bme/reports/")

    ss = pd.read_excel(f"{ddir}/{organoids['samplesheet']}", index_col=0)

    counts = CRISPRDataSet(organoids, ddir=ddir)

    # -
    samples = list(ss.index)
    palette = ss.set_index("name")["palette"].to_dict()

    # - Fold-changes
    fc = (
        counts.counts.remove_low_counts(counts.plasmids)
        .norm_rpm()
        .foldchange(counts.plasmids)
    )

    fc_gene = fc.groupby(counts.lib.reindex(fc.index)["Gene"]).mean()

    fc_gene_scaled = ReadCounts(fc_gene).scale()

    # - sgRNAs counts
    count_thres = 10

    plot_df = (
        pd.concat(
            [
                counts.counts.sum().rename("Total reads"),
                (counts.counts > count_thres).sum().rename("Guides threshold"),
                ss["name"],
            ],
            axis=1,
            sort=False,
        )
        .loc[samples]
        .dropna()
    )

    # Total counts barplot
    plt.figure(figsize=(2, 3), dpi=600)

    sns.barplot(
        x="Total reads",
        y="name",
        orient="h",
        data=plot_df.sort_values("Total reads", ascending=False),
        palette=palette,
    )

    plt.title("Total raw counts")
    plt.xlabel("Total counts (sum)")
    plt.ylabel("")

    plt.savefig(f"{dreports}/rawcounts_total_barplot.pdf", bbox_inches="tight")
    plt.close("all")

    # Guides above count threshold
    plt.figure(figsize=(2, 3), dpi=600)

    sns.barplot(
        x="Guides threshold",
        y="name",
        orient="h",
        data=plot_df.sort_values("Guides threshold", ascending=False),
        palette=palette,
    )

    plt.title(f"sgRNAs with counts > {count_thres}")
    plt.xlabel("Number of sgRNAs")
    plt.ylabel("")

    plt.savefig(f"{dreports}/rawcounts_threshold_barplot.pdf", bbox_inches="tight")
    plt.close("all")

    # - sgRNAs fold-changes
    plot_df = fc.reindex(samples, axis=1).dropna(axis=1).rename(columns=ss["name"])
    plot_df = plot_df.unstack().rename("fc").reset_index()
    plot_df.columns = ["sample", "sgRNA", "fc"]

    plt.figure(figsize=(2, 3), dpi=600)
    sns.boxplot(
        "fc",
        "sample",
        orient="h",
        data=plot_df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
    )
    plt.axvline(0, ls="-", lw=0.05, zorder=0, c="#484848")
    plt.title("sgRNA fold-changes")
    plt.savefig(f"{dreports}/foldchange_sgrna_boxplots.pdf", bbox_inches="tight")
    plt.close("all")

    # - Gene fold-changes
    plot_df = fc_gene.reindex(samples, axis=1).dropna(axis=1).rename(columns=ss["name"])
    plot_df = plot_df.unstack().rename("fc").reset_index()
    plot_df.columns = ["sample", "gene", "fc"]

    plt.figure(figsize=(2, 3), dpi=600)
    sns.boxplot(
        "fc",
        "sample",
        orient="h",
        data=plot_df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
    )
    plt.axvline(0, ls="-", lw=0.05, zorder=0, c="#484848")
    plt.title("Gene fold-changes (mean)")
    plt.savefig(f"{dreports}/foldchange_gene_boxplots.pdf", bbox_inches="tight")
    plt.close("all")

    # - Clustermap raw counts
    plot_df = (
        np.log10(counts.counts)
        .reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])
        .corr(method="spearman")
    )
    g = sns.clustermap(
        plot_df,
        cmap="Spectral",
        center=0,
        annot=True,
        fmt=".1f",
        annot_kws=dict(size=4),
        figsize=(3, 3),
        lw=0.05,
        col_colors=pd.Series(palette)[plot_df.columns],
        row_colors=pd.Series(palette)[plot_df.index],
    )
    plt.suptitle("Raw counts (log10) spearman")
    plt.savefig(f"{dreports}/clustermap_rawcounts.pdf", bbox_inches="tight")
    plt.close("all")

    # - Clustermap sgrna fold-changes
    plot_df = (
        fc.reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])
        .corr(method="spearman")
    )
    sns.clustermap(
        plot_df,
        cmap="Spectral",
        center=0,
        annot=True,
        fmt=".1f",
        annot_kws=dict(size=4),
        figsize=(3, 3),
        lw=0.05,
        col_colors=pd.Series(palette)[plot_df.columns],
        row_colors=pd.Series(palette)[plot_df.index],
    )
    plt.suptitle("sgRNAs fold-change spearman")
    plt.savefig(f"{dreports}/clustermap_foldchange_sgrna.pdf", bbox_inches="tight")
    plt.close("all")

    # - Clustermap gene fold-changes
    plot_df = (
        fc_gene.reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])
        .corr(method="spearman")
    )
    sns.clustermap(
        plot_df,
        cmap="Spectral",
        center=0,
        annot=True,
        fmt=".1f",
        annot_kws=dict(size=4),
        figsize=(3, 3),
        lw=0.05,
        col_colors=pd.Series(palette)[plot_df.columns],
        row_colors=pd.Series(palette)[plot_df.index],
    )
    plt.suptitle("Gene fold-change spearman")
    plt.savefig(f"{dreports}/clustermap_foldchange_gene.pdf", bbox_inches="tight")
    plt.close("all")

    # - Recall gene lists
    plot_df = fc_gene.reindex(samples, axis=1).dropna(axis=1).rename(columns=ss["name"])

    # Essential
    plt.figure(figsize=(2, 2), dpi=600)

    ax = plt.gca()

    _, stats_ess = QCplot.plot_cumsum_auc(
        plot_df,
        Utils.get_essential_genes(),
        palette=palette,
        legend_prop={"size": 4},
        ax=ax,
    )

    plt.savefig(f"{dreports}/qc_ess_aurc_ess.pdf", bbox_inches="tight")
    plt.close("all")

    # Non-essential
    plt.figure(figsize=(2, 2), dpi=600)

    ax = plt.gca()

    _, stats_ness = QCplot.plot_cumsum_auc(
        plot_df,
        Utils.get_non_essential_genes(),
        palette=palette,
        legend_prop={"size": 4},
        ax=ax,
    )

    plt.savefig(f"{dreports}/qc_ess_aurc_ness.pdf", bbox_inches="tight")
    plt.close("all")

    # - Scaled gene fold-changes
    # Boxplots
    plot_df = (
        fc_gene_scaled.reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])
    )
    plot_df = plot_df.unstack().rename("fc").reset_index()
    plot_df.columns = ["sample", "gene", "fc"]

    plt.figure(figsize=(2, 3), dpi=600)
    sns.boxplot(
        "fc",
        "sample",
        orient="h",
        data=plot_df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
    )
    plt.axvline(0, ls="-", lw=0.05, zorder=0, c="#484848")
    plt.title("Gene fold-changes (mean, scaled)")
    plt.savefig(f"{dreports}/foldchange_gene_scaled_boxplots.pdf", bbox_inches="tight")
    plt.close("all")

    # Clustermap
    plot_df = (
        fc_gene_scaled.reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])
        .corr(method="spearman")
    )
    sns.clustermap(
        plot_df,
        cmap="Spectral",
        center=0,
        annot=True,
        fmt=".1f",
        annot_kws=dict(size=4),
        figsize=(3, 3),
        lw=0.05,
        col_colors=pd.Series(palette)[plot_df.columns],
        row_colors=pd.Series(palette)[plot_df.index],
    )
    plt.suptitle("Scaled gene fold-change spearman")
    plt.savefig(
        f"{dreports}/clustermap_foldchange_scaled_gene.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Clustermap filtered
    plot_df = fc_gene_scaled.reindex(samples, axis=1).dropna(axis=1)
    plot_df = plot_df.loc[(fc_gene_scaled.abs() > 1).sum(1) > 5]
    plot_df = plot_df.rename(columns=ss["name"]).corr(method="spearman")

    sns.clustermap(
        plot_df,
        cmap="Spectral",
        center=0,
        annot=True,
        fmt=".1f",
        annot_kws=dict(size=4),
        figsize=(3, 3),
        lw=0.05,
        col_colors=pd.Series(palette)[plot_df.columns],
        row_colors=pd.Series(palette)[plot_df.index],
    )
    plt.suptitle("Filtered and Scaled gene fold-change spearman")
    plt.savefig(
        f"{dreports}/clustermap_foldchange_scaled_filtered_gene.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    # -
    plot_df = fc_gene_scaled.rename(columns=ss["name"])

    def triu_plot(x, y, color, label, **kwargs):
        data, x_e, y_e = np.histogram2d(x, y, bins=20)
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
        )

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

    grid = sns.PairGrid(plot_df, height=1.1, despine=False)

    grid.map_diag(diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)

    for i, j in zip(*np.tril_indices_from(grid.axes, -1)):
        ax = grid.axes[i, j]
        r, p = spearmanr(plot_df.iloc[:, i], plot_df.iloc[:, j])
        ax.annotate(
            f"R={r:.2f}\np={p:.1e}",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    grid = grid.map_upper(triu_plot, marker="o", edgecolor="", cmap="Spectral_r", s=2)

    grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(
        f"{dreports}/pairplot_gene_foldchanges_scatter.png", bbox_inches="tight"
    )
    plt.close("all")

    # Copyright (C) 2019 Emanuel Goncalves
