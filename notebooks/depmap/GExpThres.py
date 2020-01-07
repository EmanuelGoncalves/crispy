#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import crispy
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from crispy.BGExp import GExp
from crispy.QCPlot import QCplot
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import pearsonr, skewtest
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model.base import LinearRegression
from sklearn.metrics import jaccard_score, matthews_corrcoef

LOG = logging.getLogger("Crispy")

DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "depmap/reports/")

KWS_SCATTER = dict(edgecolor="w", lw=0.3, s=10, alpha=0.6)
KWS_LINE = dict(lw=1.0, color=CrispyPlot.PAL_DBGD[1], alpha=1.0)
KWS_JOINT = dict(lowess=False, scatter_kws=KWS_SCATTER, line_kws=KWS_LINE)
KWS_MARGINAL = dict(kde=False, hist_kws={"linewidth": 0})
KWS_ANNOT = dict(stat="R")


class GDSCGexp:
    GEXP_FILE = f"{DPATH}/rnaseq_voom.csv.gz"
    SAMPLESHEET_FILE = f"{DPATH}/ModelList_20191106.csv"
    GROWTH_FILE = f"{DPATH}/GrowthRates_v1.3.0_20190222.csv"
    TISSUE_PAL_FILE = f"{DPATH}/tissue_palette.csv"

    def __init__(self):
        self.growth = pd.read_csv(self.GROWTH_FILE)

        self.ss = pd.read_csv(self.SAMPLESHEET_FILE, index_col=0)
        self.ss["growth"] = self.growth.groupby("model_id")["GROWTH_RATE"].mean()

        self.gexp = pd.read_csv(self.GEXP_FILE, index_col=0)

        self.pal_tissue = pd.read_csv(self.TISSUE_PAL_FILE, index_col=0)["color"]

    @staticmethod
    def gene_lh(tcga_genes, gtex_genes, tcga_thres=(-2.5, 7.5), gtex_thres=(0.5, 2)):
        genes_low_tcga = set(tcga_genes[tcga_genes < tcga_thres[0]].index)
        genes_low_gtex = set(gtex_genes[gtex_genes < gtex_thres[0]].index)

        genes_high_tcga = set(tcga_genes[tcga_genes > tcga_thres[1]].index)
        genes_high_gtex = set(gtex_genes[gtex_genes > gtex_thres[1]].index)

        return dict(
            low=set.intersection(genes_low_tcga, genes_low_gtex),
            high=set.intersection(genes_high_tcga, genes_high_gtex),
        )


class TCGAGexp:
    GEXP_FILE = f"{DPATH}/GSE62944_merged_expression_voom.tsv"
    CANCER_TYPE_FILE = f"{DPATH}/GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt"

    def __init__(self, gene_subset=None):
        self.gexp = pd.read_csv(self.GEXP_FILE, index_col=0, sep="\t")

        if gene_subset is not None:
            self.gexp = self.gexp[self.gexp.index.isin(gene_subset)]

        self.gexp_genes = self.gexp.median(1).sort_values(ascending=False)
        self.gexp_genes_std = self.gexp.std(1).sort_values(ascending=False)
        self.gexp_genes_skew = pd.Series(
            skewtest(self.gexp.T)[0], index=self.gexp.index
        )

        self.cancer_type = pd.read_csv(
            self.CANCER_TYPE_FILE, sep="\t", header=None, index_col=0
        )[1]
        self.cancer_type = self.cancer_type.append(
            pd.Series(
                {x: "Normal" for x in self.gexp.columns if x not in self.cancer_type}
            )
        )

        colors = (
            sns.color_palette("tab20c").as_hex() + sns.color_palette("tab20b").as_hex()
        )
        self.cancer_type_palette = dict(
            zip(natsorted(self.cancer_type.value_counts().index), colors)
        )


class GTEXGexp:
    GEXP_FILE = f"{DPATH}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"

    def __init__(self, gene_subset=None):
        self.gexp = pd.read_csv(self.GEXP_FILE, sep="\t")
        self.gexp = self.gexp.drop(columns=["Name"]).groupby("Description").median()

        if gene_subset is not None:
            self.gexp = self.gexp[self.gexp.index.isin(gene_subset)]

        self.gexp_genes = np.log10(self.gexp + 1).median(1).sort_values(ascending=False)
        self.gexp_genes_std = (
            np.log10(self.gexp + 1).std(1).sort_values(ascending=False)
        )
        self.gexp_genes_skew = pd.Series(
            skewtest(np.log10(self.gexp + 1).T)[0], index=self.gexp.index
        )


def pc_labels(n):
    return [f"PC{i}" for i in np.arange(1, n + 1)]


def dim_reduction(
    df,
    pca_ncomps=50,
    tsne_ncomps=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate=200.0,
    n_iter=1000,
):
    # PCA
    df_pca = PCA(n_components=pca_ncomps).fit_transform(df.T)
    df_pca = pd.DataFrame(df_pca, index=df.T.index, columns=pc_labels(pca_ncomps))

    # tSNE
    df_tsne = TSNE(
        n_components=tsne_ncomps,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
    ).fit_transform(df_pca)
    df_tsne = pd.DataFrame(df_tsne, index=df_pca.index, columns=pc_labels(tsne_ncomps))

    return df_tsne, df_pca


def plot_dim_reduction(data, palette=None, ctype="tSNE"):
    if "tissue" not in data.columns:
        data = data.assign(tissue="All")

    if palette is None:
        palette = dict(All=CrispyPlot.PAL_DBGD[0])

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=600)

    for t, df in data.groupby("tissue"):
        ax.scatter(
            df["PC1"], df["PC2"], c=palette[t], marker="o", edgecolor="", s=5, label=t, alpha=.8
        )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.axis("off" if ctype == "tSNE" else "on")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 4},
        frameon=False,
        title="Tissue",
    ).get_title().set_fontsize("5")

    return ax


if __name__ == "__main__":
    # GDSC GExp
    #
    gdsc = GDSCGexp()

    # TCGA imports
    #
    tcga = TCGAGexp(gene_subset=set(gdsc.gexp.index))

    # GTEx gene median expression
    #
    gtex = GTEXGexp(gene_subset=set(gdsc.gexp.index))

    #
    #
    tcga_thres, gtex_thres = (-2.5, 7.5), (0.5, 2.0)

    #
    #
    pal_genes_lh = {"low": "#fc8d62", "high": "#2b8cbe"}

    genes_lh = gdsc.gene_lh(
        tcga.gexp_genes, gtex.gexp_genes, tcga_thres=tcga_thres, gtex_thres=gtex_thres
    )
    genes_lh_df = pd.DataFrame(
        [dict(gene=g, gtype=gtype) for gtype in genes_lh for g in genes_lh[gtype]]
    )
    genes_lh_df.to_csv(f"{DPATH}/GExp_genesets_20191126.csv", index=False)

    # TCGA and GTEX gene histograms
    #
    for n, df in [("TCGA", tcga.gexp_genes), ("GTEX", gtex.gexp_genes)]:
        plt.figure(figsize=(2.5, 1.5), dpi=600)
        sns.distplot(
            df,
            hist=False,
            kde_kws={"cut": 0, "shade": True},
            color=CrispyPlot.PAL_DBGD[0],
        )
        plt.title(n)
        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
        plt.xlabel(
            "Gene median (RPKM voom)" if n == "TCGA" else "Gene median (TPM log10)"
        )
        plt.savefig(
            f"{RPATH}/genes_histogram_{n}.pdf", bbox_inches="tight", transparent=True
        )
        plt.close("all")

    # TCGA dimension reduction
    #
    tcga_gexp_tsne, tcga_gexp_pca = dim_reduction(tcga.gexp)

    for ctype, df in [("tSNE", tcga_gexp_tsne), ("PCA", tcga_gexp_pca)]:
        plot_df = pd.concat(
            [df, tcga.cancer_type._set_name("tissue")], axis=1, sort=False
        ).dropna()

        ax = plot_dim_reduction(
            plot_df, ctype=ctype, palette=tcga.cancer_type_palette
        )
        ax.set_title(f"{ctype} - TCGA GExp")
        plt.savefig(
            f"{RPATH}/tcga_gexp_{ctype}.pdf", bbox_inches="tight", transparent=True
        )
        plt.close("all")

    # TCGA and GTEX gene correlation
    #
    plot_df = pd.concat(
        [
            tcga.gexp_genes.rename("TCGA_median"),
            tcga.gexp_genes_std.rename("TCGA_std"),
            tcga.gexp_genes_skew.rename("TCGA_skew"),
            gtex.gexp_genes.rename("GTEX_median"),
            gtex.gexp_genes_std.rename("GTEX_std"),
            gtex.gexp_genes_skew.rename("GTEX_skew"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    for xx, yy in [("TCGA_median", "GTEX_median")]:
        g = sns.JointGrid(x=xx, y=yy, data=plot_df, space=0)

        g = g.plot_joint(
            plt.hexbin, cmap="Spectral_r", gridsize=100, mincnt=1, bins="log", lw=0
        )
        g.ax_joint.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        x_lim, y_lim = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
        for i, n in enumerate(["low", "high"]):
            x, y = tcga_thres[i], gtex_thres[i]
            w, h = x_lim[i] - x, y_lim[i] - y

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=0,
                facecolor=CrispyPlot.PAL_DBGD[0],
                alpha=0.25,
                zorder=0,
            )
            g.ax_joint.annotate(
                f"N={len(genes_lh[n])}",
                (x + w / 2, y + h / 2),
                color="k",
                weight="bold",
                fontsize=6,
                ha="center",
                va="center",
            )
            g.ax_joint.add_patch(rect)

        g.ax_joint.set_ylim(y_lim)
        g.ax_joint.set_xlim(x_lim)

        g = g.plot_marginals(
            sns.distplot,
            kde=False,
            color=CrispyPlot.PAL_DBGD[0],
            hist_kws={"linewidth": 0},
        )
        g.set_axis_labels(
            f"{xx.replace('_', ' ')} (voom)", f"{yy.replace('_', ' ')} (TPM log10)"
        )

        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(
            f"{RPATH}/genes_TCGA_GTEX_corrplot_{xx.split('_')[1]}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

    # Mean vs Std
    #
    for n, (x, y) in [
        ("TCGA", ("TCGA_median", "TCGA_std")),
        ("GTEX", ("GTEX_median", "GTEX_std")),
    ]:
        g = sns.JointGrid(x=x, y=y, data=plot_df, space=0)

        g = g.plot_joint(
            plt.hexbin, cmap="Spectral_r", gridsize=100, mincnt=1, bins="log", lw=0
        )

        for s in ["low", "high"]:
            g.ax_joint.scatter(
                plot_df.loc[genes_lh[s], x],
                plot_df.loc[genes_lh[s], y],
                c=pal_genes_lh[s],
                marker="o",
                edgecolor="white",
                linewidth=0.1,
                s=3,
                alpha=1.0,
                label=s,
            )
        g.ax_joint.legend(frameon=False)

        g.ax_joint.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        g = g.plot_marginals(
            sns.distplot,
            kde=False,
            color=CrispyPlot.PAL_DBGD[0],
            hist_kws={"linewidth": 0},
        )

        x_label = (
            "TCGA gene median (voom)" if n == "TCGA" else "GTEX gene median (TPM log10)"
        )
        y_label = "TCGA gene std (voom)" if n == "TCGA" else "GTEX gene std (TPM log10)"
        g.set_axis_labels(x_label, y_label)

        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(
            f"{RPATH}/bgexp/genes_std_{n}.pdf", bbox_inches="tight", transparent=True
        )
        plt.close("all")

    # Genes distribution in GDSC
    #
    plt.figure(figsize=(2.5, 1.5), dpi=600)
    for s in genes_lh:
        sns.distplot(
            gdsc.gexp.loc[genes_lh[s]].median(1),
            hist=False,
            label=s,
            kde_kws={"cut": 0, "shade": True},
            color=pal_genes_lh[s],
        )
    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
    plt.xlabel("GDSC gene median (voom)")
    plt.legend(frameon=False, prop={"size": 5})
    plt.savefig(
        f"{RPATH}/genes_lh_gdsc_histograms.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    # Discretise gene-expression (calculate)
    #
    gexp = GExp(genesets=genes_lh)
    for s in gdsc.gexp:
        sample = gdsc.gexp[s]
        LOG.info(f"Sample={s}")

        sample_disc = gexp.discretise(sample, max_fpr=0.1)
        LOG.info(sample_disc[gexp.genesets].sum())

        # Export report
        sample_disc.to_csv(f"{RPATH}/bgexp/bgexp_{s}.csv")

    # GDSC discretised gene-expression (import)
    #
    gdsc_disc = pd.concat(
        [
            pd.read_csv(f"{RPATH}/bgexp/bgexp_{s}.csv", index_col=0)[
                ["high", "low"]
            ].add_suffix(f"_{s}")
            for s in gdsc.gexp
        ],
        axis=1,
    )

    # GDSC discretised log-ratio
    #
    gdsc_lr = pd.DataFrame(
        {
            s: pd.read_csv(f"{RPATH}/bgexp/bgexp_{s}.csv", index_col=0)["lr_mean"]
            for s in gdsc.gexp
        }
    )

    # Assemble discretised table
    #
    def sample_melt_df(sample):
        df = pd.read_csv(f"{RPATH}/bgexp/bgexp_{sample}.csv", index_col=0)
        df = df.query("(high != 0) | (low != 0)")[["low", "high"]]
        df = pd.melt(df.reset_index(), id_vars="index").query("value == 1").drop(columns=["value"])
        df = df.rename(columns={"variable": "dtype", "index": "gene"}).assign(sample=sample)
        return df

    gdsc_disc_table = pd.concat([sample_melt_df(s) for s in gdsc.gexp])

    gdsc_disc_table = gdsc_disc_table.assign(value=1)
    gdsc_disc_table = gdsc_disc_table.assign(name=gdsc_disc_table["gene"] + "_" + gdsc_disc_table["dtype"].values)

    gdsc_disc_table = pd.pivot_table(gdsc_disc_table, index="name", columns="sample", values="value", fill_value=0)

    gdsc_disc_table.to_csv(f"{DPATH}/GDSC_discretised_table.csv", compression="gzip")

    # Number of diff expressed genes
    #
    plot_df = pd.DataFrame(
        {
            s: {gtype: gdsc_disc[f"{gtype}_{s}"].sum() for gtype in ["low", "high"]}
            for s in gdsc.gexp
        }
    ).T.sort_values("high")
    plot_df = pd.concat([plot_df, gdsc.ss["tissue"]], axis=1, sort=False).dropna()
    plot_df = plot_df.reset_index()

    _, ax = plt.subplots(figsize=(3, 3))

    for t, df in plot_df.groupby("tissue"):
        ax.scatter(
            df["low"],
            df["high"],
            c=gdsc.pal_tissue[t],
            s=5,
            label=t,
            lw=0.1,
            edgecolor="white",
            alpha=0.7,
        )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
    ax.set_xlabel(f"Number of low expressed genes")
    ax.set_ylabel(f"Number of low expressed genes")
    ax.legend(
        frameon=False, prop={"size": 4}, loc="center left", bbox_to_anchor=(1, 0.5)
    )
    plt.savefig(
        f"{RPATH}/bgexp_highlow_scatter.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    # GDSC tSNE
    #
    gdsc_lr_tsne, gdsc_lr_pca = dim_reduction(gdsc_lr)
    gdsc_gexp_tsne, gdsc_gexp_pca = dim_reduction(gdsc.gexp)

    # GDSC tSNE plot
    decomp = dict(
        lr=dict(tSNE=gdsc_lr_tsne, PCA=gdsc_lr_pca),
        gexp=dict(tSNE=gdsc_gexp_tsne, PCA=gdsc_gexp_pca),
    )

    for dtype in decomp:
        for ctype in decomp[dtype]:
            plot_df = pd.concat(
                [decomp[dtype][ctype], gdsc.ss["tissue"]], axis=1, sort=False
            ).dropna()

            plt.figure(figsize=(4, 4))
            for t, df in plot_df.groupby("tissue"):
                plt.scatter(
                    df["PC1"],
                    df["PC2"],
                    c=gdsc.pal_tissue[t],
                    marker="o",
                    edgecolor="",
                    s=5,
                    label=t,
                )
            plt.title(f"{ctype} - GDSC {dtype}")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.axis("off" if ctype == "tSNE" else "on")
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                prop={"size": 4},
                frameon=False,
                title="Tissue",
            ).get_title().set_fontsize("5")
            plt.savefig(
                f"{RPATH}/gdsc_{dtype}_{ctype}.pdf",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close("all")

    # Clustering discretised data
    gdsc_disc_dict = gdsc_disc.loc[gdsc_disc.sum(1) > 10]
    gdsc_disc_dict = {
        s: gdsc_disc_dict[f"low_{s}"].replace(1, -1) + gdsc_disc_dict[f"high_{s}"]
        for s in gdsc.gexp
    }
    samples = list(gdsc.gexp)

    gdsc_mcc = {s1: {s2: np.nan for s2 in samples} for s1 in samples}
    for s1 in samples:
        LOG.info(s1)
        for s2 in samples:
            if not np.isnan(gdsc_mcc[s2][s1]):
                j = gdsc_mcc[s2][s1]

            else:
                j = matthews_corrcoef(gdsc_disc_dict[s1], gdsc_disc_dict[s2])

            gdsc_mcc[s1][s2] = j
    gdsc_mcc = pd.DataFrame(gdsc_mcc).loc[samples, samples]
    gdsc_mcc.round(5).to_csv(f"{RPATH}/bgexp_mcc_matrix.csv")

    samples = gdsc.ss.loc[samples, "tissue"].dropna()

    plot_df = gdsc_mcc.loc[samples.index, samples.index]
    sns.clustermap(
        plot_df,
        cmap="Spectral",
        annot=False,
        center=0,
        figsize=(6, 6),
        xticklabels=False,
        yticklabels=False,
        col_colors=gdsc.pal_tissue[samples.loc[plot_df.columns]].values,
        row_colors=gdsc.pal_tissue[samples.loc[plot_df.index]].values,
    )
    plt.suptitle("Discretised gene-expression")
    plt.savefig(f"{RPATH}/gdsc_disc_clustermap.png", bbox_inches="tight", dpi=300)
    plt.close("all")

    #
    #
    gdsc_low_skew = pd.Series(skewtest(gdsc.gexp.loc[genes_lh["high"]], 1)[0], index=genes_lh["high"])
    tcga_low_skew = pd.Series(skewtest(tcga.gexp.loc[genes_lh["high"]], 1)[0], index=genes_lh["high"])

    plot_df = pd.concat([
        gdsc_low_skew.rename("GDSC"), tcga_low_skew.rename("TCGA")
    ], axis=1, sort=False)

    plt.scatter(plot_df["GDSC"], plot_df["TCGA"])
    plt.show()

    # Example
    #
    s = "SIDM00424"

    sample = gdsc.gexp[s]
    sample_disc = gexp.discretise(sample, max_fpr=0.1, genesets=genes_lh, verbose=2)
    LOG.info(f"Sample low={sample_disc['low'].sum()}; Sample high={sample_disc['high'].sum()}")

    # Sample voom histogram
    ax = gexp.plot_histogram(sample)
    ax.set_title(s)
    ax.set_xlabel("Gene median (voom)")
    plt.savefig(f"{RPATH}/bgexp_{s}_hist.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")

    # Sample log-ratio tests histogram

    ax = gexp.plot_histogram(sample_disc["lr_mean"])
    ax.set_title(s)
    ax.set_xlabel("Log-ratio high / low")

    for n in gexp.genesets:
        ax.axvline(
            sample_disc[f"{n}_thres_mean"].iloc[0],
            ls="-",
            lw=0.5,
            alpha=1.0,
            zorder=0,
            c=gexp.palette[n],
        )

    plt.savefig(f"{RPATH}/bgexp_{s}_lr_hist.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")
