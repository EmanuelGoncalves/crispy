#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from scipy.stats import ks_2samp, spearmanr


def sgrna_mean_correlation(fc, lib, method="pearson"):
    glib = lib.dropna(subset=["Symbol"])

    sgrna_corr = []
    for g, df in glib.groupby("Symbol"):
        print(f"Correlating {g}")
        g_fc = fc.reindex(df.index).mean()

        sg_corrs = fc.loc[df.index].T.apply(
            lambda col: col.corr(g_fc, method=method), axis=0
        )

        sgrna_corr.append(sg_corrs)
    sgrna_corr = pd.concat(sgrna_corr)

    return sgrna_corr


if __name__ == "__main__":

    rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

    # - JACKS
    jacks = pd.read_csv(
        "/Users/eg14/Downloads/whitehead_jacks_results/jacks_29cNov2018_whitehead_grna_JACKS_results.txt",
        sep="\t",
        index_col=0,
    )

    # - Data-set
    counts = CRISPRDataSet("Sabatini_Lander_AML")

    fc = (
        counts.counts.remove_low_counts(counts.plasmids)
        .norm_rpm()
        .foldchange(counts.plasmids)
    )

    fc_gene = fc.groupby(counts.lib.reindex(fc.index)["Symbol"]).mean()

    # -
    _, original_stats = QCplot.plot_cumsum_auc(fc_gene, Utils.get_essential_genes())
    plt.show()

    # -
    sgrna_corr = sgrna_mean_correlation(fc, counts.lib)
    sgrna_corr_spearman = sgrna_mean_correlation(fc, counts.lib, method="spearman")

    # - Sets of sgRNAs
    sgrnas_essential = Utils.get_sanger_essential()
    sgrnas_essential = set(
        sgrnas_essential[sgrnas_essential["ADM_PanCancer_CF"]]["Gene"]
    )
    sgrnas_essential = set(
        counts.lib[counts.lib["Symbol"].isin(sgrnas_essential)].index
    )

    sgrnas_nonessential = Utils.get_non_essential_genes(return_series=False)
    sgrnas_nonessential = set(
        counts.lib[counts.lib["Symbol"].isin(sgrnas_nonessential)].index
    )

    sgrnas_control = {i for i in counts.lib.index if i.startswith("CTRL0")}

    sgrnas_intergenic = {i for i in counts.lib.index if i.startswith("INTERGENIC")}

    # -
    conditions = dict(
        control=dict(color="#31a354", sgrnas=sgrnas_control),
        essential=dict(color="#e6550d", sgrnas=sgrnas_essential),
        nonessential=dict(color="#3182bd", sgrnas=sgrnas_nonessential),
        intergenic=dict(color="#8c6d31", sgrnas=sgrnas_intergenic),
    )

    # -
    plt.figure(figsize=(2.5, 2), dpi=600)

    for c in conditions:
        sns.distplot(
            fc.reindex(conditions[c]["sgrnas"]).median(1).dropna(),
            hist=False,
            label=c,
            kde_kws={"cut": 0},
            color=conditions[c]["color"],
        )

    plt.xlabel("Fold-change")
    plt.legend(frameon=False)

    plt.savefig(
        f"{rpath}/sabatini_sgrnas_fc_distplot.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # -
    sgrnas_control_fc = fc.reindex(sgrnas_control).median(1).dropna()
    sgrnas_intergenic_fc = fc.reindex(sgrnas_intergenic).median(1).dropna()

    ks_sgrna = []
    for sgrna in fc.index:
        d_ctrl, p_ctrl = ks_2samp(fc.loc[sgrna], sgrnas_control_fc)
        d_inter, p_inter = ks_2samp(fc.loc[sgrna], sgrnas_intergenic_fc)

        ks_sgrna.append(
            {
                "sgrna": sgrna,
                "ks_ctrl": d_ctrl,
                "pval_ctrl": p_ctrl,
                "ks_intergenic": d_inter,
                "pval_intergenic": p_inter,
            }
        )

    ks_sgrna = pd.DataFrame(ks_sgrna).set_index("sgrna")
    ks_sgrna["median_fc"] = fc.loc[ks_sgrna.index].median(1).values
    ks_sgrna["jacks"] = jacks.loc[ks_sgrna.index, "X1"]
    ks_sgrna["corr"] = sgrna_corr.reindex(ks_sgrna.index)
    ks_sgrna["corr_spearman"] = sgrna_corr_spearman.reindex(ks_sgrna.index)

    # -
    x, y = "ks_ctrl", "jacks"
    df = ks_sgrna.dropna(subset=[x, y])

    data, x_e, y_e = np.histogram2d(df[x], df[y], bins=20)
    density = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([df[x], df[y]]).T,
        method="splinef2d",
        bounds_error=False,
    )

    plt.figure(figsize=(2.5, 2), dpi=600)
    g = plt.scatter(
        df[x],
        df[y],
        c=density,
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=1,
        alpha=0.85,
    )
    plt.colorbar(g, spacing="uniform", extend="max")

    plt.axhline(0, ls=":", lw=.1, c="k", zorder=1)
    plt.axhline(1, ls="-", lw=.2, c="k", zorder=1)
    plt.axhline(2, ls=":", lw=.1, c="k", zorder=1)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(f"{rpath}/sabatini_{x}_{y}.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    # -
    x, y, z = "ks_ctrl", "jacks", "median_fc"
    df = ks_sgrna.dropna(subset=[x, y, z])

    plt.figure(figsize=(2.5, 2), dpi=600)
    g = plt.scatter(
        df[x],
        df[y],
        c=df[z],
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=1,
        alpha=0.85,
    )
    cbar = plt.colorbar(g, spacing="uniform", extend="max")
    cbar.ax.set_ylabel(z)

    plt.axhline(0, ls=":", lw=.1, c="k", zorder=1)
    plt.axhline(1, ls="-", lw=.2, c="k", zorder=1)
    plt.axhline(2, ls=":", lw=.1, c="k", zorder=1)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(f"{rpath}/sabatini_{x}_{y}_{z}.png", bbox_inches="tight", dpi=600)
    plt.close("all")


    # -
    x, y = "ks_ctrl", "ks_intergenic"

    data, x_e, y_e = np.histogram2d(ks_sgrna[x], ks_sgrna[y], bins=20)
    density = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([ks_sgrna[x], ks_sgrna[y]]).T,
        method="splinef2d",
        bounds_error=False,
    )

    z_metrics = [
        ("median_fc", ks_sgrna["median_fc"]),
        ("density", density),
        ("jacks", ks_sgrna["jacks"]),
        ("corr", ks_sgrna["corr"]),
    ]

    for n, z_metric in z_metrics:
        plt.figure(figsize=(2.5, 2), dpi=600)
        g = plt.scatter(
            ks_sgrna[x],
            ks_sgrna[y],
            c=z_metric,
            marker="o",
            edgecolor="",
            cmap="Spectral_r",
            s=1,
            alpha=0.85,
        )
        plt.colorbar(g, spacing="uniform", extend="max")
        plt.xlabel("Control (K-S statistic)")
        plt.ylabel("Intergenic (K-S statistic)")
        plt.title(n)
        plt.savefig(
            f"{rpath}/sabatini_ks_distplot_{n}.png", bbox_inches="tight", dpi=600
        )
        plt.close("all")

    # -
    for c in conditions:
        grid = sns.JointGrid(x, y, data=ks_sgrna, space=0)

        grid.ax_joint.scatter(
            x=ks_sgrna[x],
            y=ks_sgrna[y],
            edgecolor="w",
            lw=0.05,
            s=5,
            color=CrispyPlot.PAL_DBGD[2],
            alpha=0.3,
        )

        grid.ax_joint.scatter(
            x=ks_sgrna.loc[conditions[c]["sgrnas"], x],
            y=ks_sgrna.loc[conditions[c]["sgrnas"], y],
            edgecolor="w",
            lw=0.05,
            s=2.5,
            color=conditions[c]["color"],
            alpha=0.7,
            label=c,
        )

        sns.distplot(
            ks_sgrna.loc[conditions[c]["sgrnas"], x],
            hist=False,
            kde_kws=dict(cut=0, shade=True),
            color=conditions[c]["color"],
            vertical=False,
            ax=grid.ax_marg_x,
        )
        grid.ax_marg_x.set_xlabel("")
        grid.ax_marg_x.set_ylabel("")

        sns.distplot(
            ks_sgrna.loc[conditions[c]["sgrnas"], y],
            hist=False,
            kde_kws=dict(cut=0, shade=True),
            color=conditions[c]["color"],
            vertical=True,
            ax=grid.ax_marg_y,
            label=c,
        )
        grid.ax_marg_y.set_xlabel("")
        grid.ax_marg_y.set_ylabel("")

        grid.ax_marg_y.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        plt.gcf().set_size_inches(2, 2)
        plt.savefig(
            f"{rpath}/sabatini_ks_jointplot_{c}.png", bbox_inches="tight", dpi=600
        )
        plt.close("all")

    # -
    plt.figure(figsize=(2.5, 2), dpi=600)

    sns.distplot(
        ks_sgrna["corr"].dropna(),
        hist=False,
        label="all",
        kde_kws={"cut": 0},
        color=CrispyPlot.PAL_DBGD[2],
    )

    for c in ["essential", "nonessential"]:
        sns.distplot(
            ks_sgrna.reindex(conditions[c]["sgrnas"])["corr"].dropna(),
            hist=False,
            label=c,
            kde_kws={"cut": 0},
            color=conditions[c]["color"],
        )

    plt.xlabel("Fold-change")
    plt.legend(frameon=False)

    plt.savefig(
        f"{rpath}/sabatini_sgrnas_corr_distplot.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # -
    plt.figure(figsize=(2.5, 2), dpi=600)

    sns.distplot(
        jacks["X1"].dropna(),
        hist=False,
        label="All",
        kde_kws={"cut": 0},
        color=CrispyPlot.PAL_DBGD[2],
    )

    for c in ["essential", "nonessential", "control", "intergenic"]:
        sns.distplot(
            jacks.reindex(conditions[c]["sgrnas"])["X1"].dropna(),
            hist=False,
            label=c,
            kde_kws={"cut": 0},
            color=conditions[c]["color"],
        )

    plt.xlabel("JACKS scores")
    plt.legend(frameon=False)

    plt.savefig(
        f"{rpath}/sabatini_sgrnas_jacks_distplot.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # -
    samples = list(fc)

    guides = counts.lib.loc[list(ks_sgrna.index)]

    nguides_thres = 7
    nrandomisations = 10

    gene_nguides = guides["Symbol"].value_counts()
    guides = guides[~guides["Symbol"].isin(gene_nguides[gene_nguides < nguides_thres].index)]

    # Random
    r_scores = []
    for n_guides in range(1, (nguides_thres + 1)):
        print(f"Number of guides: {n_guides}")

        for i in range(nrandomisations):
            r_guides = guides.groupby("Symbol").apply(lambda x: x.sample(n=n_guides))

            r_guides_fc = fc.loc[r_guides.drop("Symbol", axis=1).reset_index()["sgRNA_ID"].values, samples]
            r_genes_fc = r_guides_fc.groupby(counts.lib.loc[r_guides_fc.index, "Symbol"]).mean()

            _, r_ess_aurc = QCplot.plot_cumsum_auc(r_genes_fc, Utils.get_essential_genes())
            plt.close("all")

            res = pd.DataFrame(r_ess_aurc).reset_index().assign(n_guides=n_guides)

            r_scores.append(res)

    r_scores = pd.concat(r_scores).reset_index(drop=True).assign(dtype="random")

    # JACKS
    j_metric = abs(jacks.loc[guides.index, "X1"] - 1).sort_values(ascending=True)

    j_scores = []
    for n_guides in range(1, (nguides_thres + 1)):
        print(f"Number of guides: {n_guides}")

        j_guides = guides.loc[j_metric.index].groupby("Symbol").head(n=n_guides)

        j_guides_fc = fc.loc[j_guides.index, samples]
        j_genes_fc = j_guides_fc.groupby(counts.lib.loc[j_guides_fc.index, "Symbol"]).mean()
        print(f"Genes {j_genes_fc.shape}")

        # _, j_ess_aurc = QCplot.plot_cumsum_auc(j_genes_fc, Utils.get_essential_genes())
        # plt.close("all")
        j_ess_aurc = dict(auc={c: QCplot.pr_curve(j_genes_fc[c]) for c in j_genes_fc})

        j_res = pd.DataFrame(j_ess_aurc).reset_index().assign(n_guides=n_guides)

        j_scores.append(j_res)

    j_scores = pd.concat(j_scores).reset_index(drop=True).assign(dtype="jacks")

    # KI
    ki_metric = ks_sgrna.loc[guides.index, "ks_intergenic"].sort_values(ascending=False)

    ki_scores = []
    for n_guides in range(1, (nguides_thres + 1)):
        print(f"Number of guides: {n_guides}")

        ki_guides = guides.loc[ki_metric.index].groupby("Symbol").head(n=n_guides)

        ki_guides_fc = fc.loc[ki_guides.index, samples]
        ki_genes_fc = ki_guides_fc.groupby(counts.lib.loc[ki_guides_fc.index, "Symbol"]).mean()
        print(f"Genes {ki_genes_fc.shape}")

        # _, ki_ess_aurc = QCplot.plot_cumsum_auc(ki_genes_fc, Utils.get_essential_genes())
        # plt.close("all")
        ki_ess_aurc = dict(auc={c: QCplot.pr_curve(ki_genes_fc[c]) for c in ki_genes_fc})

        ki_res = pd.DataFrame(ki_ess_aurc).reset_index().assign(n_guides=n_guides)

        ki_scores.append(ki_res)

    ki_scores = pd.concat(ki_scores).reset_index(drop=True).assign(dtype="ki")

    # KS
    k_metric = ks_sgrna.loc[guides.index, "ks_ctrl"].sort_values(ascending=False)

    k_scores = []
    for n_guides in range(1, (nguides_thres + 1)):
        print(f"Number of guides: {n_guides}")

        k_guides = guides.loc[k_metric.index].groupby("Symbol").head(n=n_guides)

        k_guides_fc = fc.loc[k_guides.index, samples]
        k_genes_fc = k_guides_fc.groupby(counts.lib.loc[k_guides_fc.index, "Symbol"]).mean()
        print(f"Genes {k_genes_fc.shape}")

        # _, k_ess_aurc = QCplot.plot_cumsum_auc(k_genes_fc, Utils.get_essential_genes())
        # plt.close("all")
        k_ess_aurc = dict(auc={c: QCplot.pr_curve(k_genes_fc[c]) for c in k_genes_fc})

        k_res = pd.DataFrame(k_ess_aurc).reset_index().assign(n_guides=n_guides)

        k_scores.append(k_res)

    k_scores = pd.concat(k_scores).reset_index(drop=True).assign(dtype="ks")

    # Combined
    kj_metric = ks_sgrna.apply(lambda v: v["ks_intergenic"] if 0 < v["jacks"] < 2 else v["ks_intergenic"] - 0.5, axis=1)
    kj_metric = kj_metric.loc[guides.index].sort_values(ascending=False)

    kj_scores = []
    for n_guides in range(1, (nguides_thres + 1)):
        print(f"Number of guides: {n_guides}")

        kj_guides = guides.loc[kj_metric.index].groupby("Symbol").head(n=n_guides)

        kj_guides_fc = fc.loc[kj_guides.index, samples]
        kj_genes_fc = kj_guides_fc.groupby(counts.lib.loc[kj_guides_fc.index, "Symbol"]).mean()
        print(f"Genes {kj_genes_fc.shape}")

        # _, kj_ess_aurc = QCplot.plot_cumsum_auc(kj_genes_fc, Utils.get_essential_genes())
        # plt.close("all")
        kj_ess_aurc = dict(auc={c: QCplot.pr_curve(kj_genes_fc[c]) for c in kj_genes_fc})

        kj_res = pd.DataFrame(kj_ess_aurc).reset_index().assign(n_guides=n_guides)

        kj_scores.append(kj_res)

    kj_scores = pd.concat(kj_scores).reset_index(drop=True).assign(dtype="combined")

    # -
    # Random
    plt.figure(figsize=(2, 2), dpi=600)

    sns.barplot(x="n_guides", y="auc", data=r_scores, color=CrispyPlot.PAL_DBGD[0])

    plt.xlabel("Number of guides (random)")
    plt.ylabel("Recall essential genes")

    plt.ylim(0.7)

    plt.savefig(
        f"{rpath}/sabatini_random_guides_ess_aurc.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # JACKS
    plt.figure(figsize=(2, 2), dpi=600)

    sns.barplot(x="n_guides", y="auc", data=j_scores, color=CrispyPlot.PAL_DBGD[0])

    plt.xlabel("Number of guides (top JACKS)")
    plt.ylabel("Recall essential genes")

    plt.ylim(0.7)

    plt.savefig(
        f"{rpath}/sabatini_topjacks_guides_ess_aurc.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # KS
    plt.figure(figsize=(2, 2), dpi=600)

    sns.barplot(x="n_guides", y="auc", data=k_scores, color=CrispyPlot.PAL_DBGD[0])

    plt.xlabel("Number of guides (top KS)")
    plt.ylabel("Recall essential genes")

    plt.ylim(0.7)

    plt.savefig(
        f"{rpath}/sabatini_topks_guides_ess_aurc.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # KI
    plt.figure(figsize=(2, 2), dpi=600)

    sns.barplot(x="n_guides", y="auc", data=ki_scores, color=CrispyPlot.PAL_DBGD[0])

    plt.xlabel("Number of guides (top KI)")
    plt.ylabel("Recall essential genes")

    plt.ylim(0.7)

    plt.savefig(
        f"{rpath}/sabatini_topki_guides_ess_aurc.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # KS + JACKS
    plt.figure(figsize=(2, 2), dpi=600)

    sns.barplot(x="n_guides", y="auc", data=kj_scores, color=CrispyPlot.PAL_DBGD[0])

    plt.xlabel("Number of guides (top KS + JACKS)")
    plt.ylabel("Recall essential genes")

    plt.ylim(0.7)

    plt.savefig(
        f"{rpath}/sabatini_topks+jacks_guides_ess_aurc.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # -
    plot_df = pd.concat([
        k_scores.set_index(["index", "n_guides"]).add_prefix("k_"),
        j_scores.set_index(["index", "n_guides"]).add_prefix("j_"),
        kj_scores.set_index(["index", "n_guides"]).add_prefix("kj_"),
        ki_scores.set_index(["index", "n_guides"]).add_prefix("ki_"),
    ], axis=1, sort=False).reset_index()

    plt.figure(figsize=(2.2, 2), dpi=600)

    sns.scatterplot(plot_df["ki_auc"], plot_df["j_auc"], plot_df["n_guides"], legend="full", s=8, lw=0, palette="Spectral_r")

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, 'k-', lw=.3, zorder=0)

    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5), title=None)
    plt.savefig(
        f"{rpath}/sabatini_sample_aucs_scatter.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")
