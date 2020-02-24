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
        read_counts="rawcounts.csv.gz",
        library="Yusa_v1.1.csv.gz",
        plasmids=["Plasmid_v1.1"],
        samplesheet="samplesheet.xlsx",
        exclude_samples=[
            "EGAN00002143461.sample",
            "EGAN00002143462.sample",
            "EGAN00002143463.sample",
            "EGAN00002143464.sample",
            "EGAN00002143465.sample",
            "EGAN00002143466.sample",
        ],
    )

    # - Imports
    ddir = pkg_resources.resource_filename("data", "organoids/bme2/")
    dreports = pkg_resources.resource_filename("notebooks", "bme/reports/")

    ss = pd.read_excel(f"{ddir}/{organoids['samplesheet']}", index_col=1).query(
        "organoid == 'COLO-027'"
    )

    counts = CRISPRDataSet(organoids, ddir=ddir)

    # -
    samples = list(set(ss.index).intersection(ss.index))
    palette = ss.set_index("name")["palette"]

    # - Fold-changes
    fc = (
        counts.counts.remove_low_counts(counts.plasmids)
        .norm_rpm()
        .foldchange(counts.plasmids)
    )

    fc_gene = fc.groupby(counts.lib.reindex(fc.index)["Gene"]).mean()

    fc_gene_scaled = ReadCounts(fc_gene).scale()

    # -
    fc_gene_scaled.rename(columns=ss["name"]).round(5).to_excel(
        f"{dreports}/gene_scaled_fold_changes.xlsx"
    )
    fc_gene_scaled.rename(columns=ss["name"]).corr(method="spearman").to_excel(
        f"{dreports}/samples_correlation_matrix.xlsx"
    )

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

    for x in ["Total reads", "Guides threshold"]:
        plt.figure(figsize=(2, 1.5), dpi=600)

        sns.barplot(
            x=x,
            y="name",
            orient="h",
            linewidth=0,
            saturation=1,
            order=palette.index,
            data=plot_df.sort_values("Total reads", ascending=False),
            palette=palette,
        )
        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
        plt.title(
            "Total raw counts"
            if x == "Total reads"
            else f"sgRNAs with counts > {count_thres}"
        )
        plt.xlabel("Total counts (sum)" if x == "Total reads" else "Number of sgRNAs")
        plt.ylabel("")

        plt.savefig(
            f"{dreports}/rawcounts_{x.replace(' ', '_')}_barplot.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

    # - Fold-changes boxplots
    for n, df in [("sgRNA fold-change", fc), ("Gene fold-change", fc_gene)]:
        plot_df = df.reindex(samples, axis=1).dropna(axis=1).rename(columns=ss["name"])
        plot_df = plot_df.unstack().rename("fc").reset_index()
        plot_df.columns = ["sample", "sgRNA", "fc"]

        plt.figure(figsize=(2, 1.5), dpi=600)
        sns.boxplot(
            "fc",
            "sample",
            orient="h",
            data=plot_df,
            order=palette.index,
            palette=palette,
            saturation=1,
            showcaps=False,
            flierprops=CrispyPlot.FLIERPROPS,
        )
        plt.xlabel("Fold-change (log2)")
        plt.ylabel("")
        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
        plt.title(n)
        plt.savefig(
            f"{dreports}/boxplots_{n.replace(' ', '_')}.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # - Fold-changes clustermap
    for n, df in [("sgRNA fold-change", fc), ("Gene fold-change", fc_gene)]:
        plot_df = (
            df.reindex(samples, axis=1)
            .dropna(axis=1)
            .rename(columns=ss["name"])
            .corr(method="spearman")
        )
        sns.clustermap(
            plot_df,
            cmap="Spectral",
            annot=True,
            center=0,
            fmt=".2f",
            annot_kws=dict(size=4),
            figsize=(3, 3),
            lw=0.05,
            col_colors=pd.Series(palette)[plot_df.columns].rename("BME"),
            row_colors=pd.Series(palette)[plot_df.index].rename("BME"),
        )
        plt.suptitle(n)
        plt.savefig(
            f"{dreports}/clustermap_{n.replace(' ', '_')}.pdf", bbox_inches="tight"
        )
        plt.close("all")

    # - Recall gene lists
    plot_df = (
        fc_gene.reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])[palette.index]
    )

    for n, gset in [
        ("Essential", Utils.get_essential_genes()),
        ("Non-essential", Utils.get_non_essential_genes()),
    ]:
        # Aroc
        plt.figure(figsize=(2, 2), dpi=600)
        ax = plt.gca()
        _, stats_ess = QCplot.plot_cumsum_auc(
            plot_df, gset, palette=palette, legend_prop={"size": 4}, ax=ax
        )
        plt.title(f"{n} recall curve")
        plt.xlabel("Percent-rank of genes")
        plt.ylabel("Cumulative fraction")
        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
        plt.savefig(f"{dreports}/roccurves_{n}.pdf", bbox_inches="tight")
        plt.close("all")

        # Barplot
        df = pd.Series(stats_ess["auc"])[palette.index].rename("auc").reset_index()

        plt.figure(figsize=(2, 1), dpi=600)
        sns.barplot(
            "auc",
            "name",
            data=df,
            palette=palette,
            linewidth=0,
            saturation=1,
            orient="h",
        )
        plt.axvline(0.5, ls="-", lw=0.1, alpha=1.0, zorder=0, color="k")
        plt.xlabel("Area under the recall curve")
        plt.ylabel("")
        plt.title(n)
        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
        plt.savefig(f"{dreports}/roccurves_{n}_barplot.pdf", bbox_inches="tight")
        plt.close("all")

    # - Scatter grid
    plot_df = (
        fc_gene.reindex(samples, axis=1)
        .dropna(axis=1)
        .rename(columns=ss["name"])[palette.index]
    )
    plot_df.columns = [c.replace("COLO-027 ", "") for c in plot_df.columns]

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
            f"R={r:.2f}\np={p:.1e}" if p != 0 else f"R={r:.2f}\np<0.0001",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    grid = grid.map_upper(triu_plot, marker="o", edgecolor="", cmap="Spectral_r", s=2)
    grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle("COLO-027", y=1.02)
    plt.gcf().set_size_inches(6, 6)
    plt.savefig(
        f"{dreports}/pairplot_gene_fold_changes.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # - Scatter grid (replicates averaged)
    plot_df.columns = [" ".join(c.split(" ")[:2]) for c in plot_df.columns]
    plot_df = plot_df.groupby(plot_df.columns, axis=1).mean()
    plot_df["density"] = CrispyPlot.density_interpolate(
        plot_df["BME 5%"], plot_df["BME 80%"]
    )

    x_min, x_max = (
        plot_df[["BME 5%", "BME 80%"]].min().min(),
        plot_df[["BME 5%", "BME 80%"]].max().max(),
    )

    r, p = spearmanr(plot_df["BME 5%"], plot_df["BME 80%"])
    rannot = f"R={r:.2f}; p={p:.1e}" if p != 0 else f"R={r:.2f}; p<0.0001"

    ax = plt.gca()

    ax.scatter(
        plot_df["BME 5%"],
        plot_df["BME 80%"],
        c=plot_df["density"],
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=2,
    )
    ax.annotate(
        rannot,
        xy=(0.95, 0.05),
        xycoords=ax.transAxes,
        ha="right", va="center", fontsize=7,
    )
    ax.set_xlabel("BME 5%\ngene log2 fold-change")
    ax.set_ylabel("BME 80%\ngene log2 fold-change")
    ax.set_title("COLO-027")
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(
        f"{dreports}/pairplot_average_gene_fold_changes.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    # - Waterfall plot BME 5%
    for c in ["BME 5%", "BME 80%"]:
        plot_df = plot_df.sort_values(c)
        plot_df["index"] = np.arange(plot_df.shape[0])

        genes_highlight = ["WRN", "BRAF"]
        genes_palette = sns.color_palette("Set2", n_colors=len(genes_highlight)).as_hex()

        plt.figure(figsize=(3, 2), dpi=600)
        plt.scatter(plot_df["index"], plot_df[c], c=CrispyPlot.PAL_DBGD[2], s=5, linewidths=0)

        for i, g in enumerate(genes_highlight):
            plt.scatter(plot_df.loc[g, "index"], plot_df.loc[g, c], c=genes_palette[i], s=10, linewidths=0, label=g)

        q10_fc = plot_df[c].quantile(.1)
        plt.axhline(q10_fc, ls="--", color="k", lw=.1, zorder=0)
        plt.text(-5, q10_fc, "Top 10%", color="k", ha='left', va='bottom', fontsize=5)

        plt.xlabel("Rank of genes")
        plt.ylabel(f"{c}\ngene log2 fold-change")
        plt.title("COLO-027")
        plt.legend(frameon=False)

        plt.axhline(0, ls="-", color="k", lw=.3, zorder=0)
        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="both")

        plt.savefig(
            f"{dreports}/waterfall_{c}.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

    # Copyright (C) 2019 Emanuel Goncalves
