#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
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

        sg_corrs = fc.loc[df.index].T.apply(lambda col: col.corr(g_fc, method=method), axis=0)

        sgrna_corr.append(sg_corrs)
    sgrna_corr = pd.concat(sgrna_corr)

    return sgrna_corr


if __name__ == '__main__':

    rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

    # - JACKS
    jacks = pd.read_csv("/Users/eg14/Downloads/whitehead_jacks_results/jacks_29cNov2018_whitehead_grna_JACKS_results.txt", sep="\t", index_col=0)

    # - Data-set
    counts = CRISPRDataSet("Sabatini_Lander_AML")

    fc = (
        counts.counts.remove_low_counts(counts.plasmids)
        .norm_rpm()
        .foldchange(counts.plasmids)
    )

    # -
    sgrna_corr = sgrna_mean_correlation(fc, counts.lib)
    sgrna_corr_spearman = sgrna_mean_correlation(fc, counts.lib, method="spearman")

    # - Sets of sgRNAs
    sgrnas_essential = Utils.get_sanger_essential()
    sgrnas_essential = set(sgrnas_essential[sgrnas_essential["ADM_PanCancer_CF"]]["Gene"])
    sgrnas_essential = set(counts.lib[counts.lib["Symbol"].isin(sgrnas_essential)].index)

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

    plt.savefig(f"{rpath}/sabatini_sgrnas_fc_distplot.png", bbox_inches="tight", dpi=600)
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
    x, y = "corr_spearman", "median_fc"
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
    plt.xlabel("sgRNA spearman mean gene")
    plt.ylabel("sgRNA median FC")
    plt.savefig(f"{rpath}/sabatini_spearman_medianfc.png", bbox_inches="tight", dpi=600)
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
        plt.savefig(f"{rpath}/sabatini_ks_distplot_{n}.png", bbox_inches="tight", dpi=600)
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
        plt.savefig(f"{rpath}/sabatini_ks_jointplot_{c}.png", bbox_inches="tight", dpi=600)
        plt.close("all")


    # -
    plt.figure(figsize=(2.5, 2), dpi=600)

    sns.distplot(
        ks_sgrna["corr"].dropna(),
        hist=False,
        label=c,
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

    plt.savefig(f"{rpath}/sabatini_sgrnas_corr_distplot.png", bbox_inches="tight", dpi=600)
    plt.close("all")
