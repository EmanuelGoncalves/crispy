import os
import gzip
import logging
import numpy as np
import pandas as pd
import pkg_resources
from Bio import SeqIO
from Bio import pairwise2
from Bio.SeqUtils import GC
from adjustText import adjust_text
import seaborn as sns
import matplotlib as mpl
from natsort import natsorted
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from crispy import logger as LOG
from GuideDesign import GuideDesign
from crispy.CRISPRData import ReadCounts
from crispy import CrispyPlot, QCplot, Utils
from crispy.LibRepresentationReport import LibraryRepresentaion


DPATH = pkg_resources.resource_filename("data", "dualguide/")
RPATH = pkg_resources.resource_filename("notebooks", "dualguide/reports/")


def pairgrid(df, annotate_guides=False, ntop_annotate=3):
    def triu_plot(x, y, color, label, **kwargs):
        ax = plt.gca()

        z = CrispyPlot.density_interpolate(x, y)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, **kwargs)

        ax.axhline(0, ls=":", lw=0.1, c="#484848", zorder=0)
        ax.axvline(0, ls=":", lw=0.1, c="#484848", zorder=0)

        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, ls=":", lw=0.1, c="#484848", zorder=0)

        if annotate_guides:
            diff = (x - y).sort_values()
            idxs = list(diff.head(ntop_annotate).index) + list(diff.tail(ntop_annotate).index)

            texts = [
                ax.text(
                    x.loc[i],
                    y.loc[i],
                    ";".join(lib.loc[i, ["gene1", "gene2"]]),
                    color="k",
                    fontsize=4,
                )
                for i in idxs
            ]
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
                ax=ax,
            )

    def diag_plot(x, color, label, **kwargs):
        sns.distplot(x, label=label)

    grid = sns.PairGrid(df, height=1.1, despine=False)

    grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.gcf().set_size_inches(2 * df.shape[1], 2 * df.shape[1])

    grid.map_diag(diag_plot, kde=True, hist_kws=dict(linewidth=0), bins=30)

    for i, j in zip(*np.tril_indices_from(grid.axes, -1)):
        ax = grid.axes[i, j]
        r, p = spearmanr(df.iloc[:, i], df.iloc[:, j])
        ax.annotate(
            f"R={r:.2f}\np={p:.1e}" if p != 0 else f"R={r:.2f}\np<0.0001",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    grid.map_upper(triu_plot, marker="o", edgecolor="", cmap="Spectral_r", s=2)


def scattergrid(
    df,
    columns,
    xprefix,
    yprefix,
    density=True,
    highlight_text=None,
    xlabel="",
    ylabel="",
    add_corr=True,
    n_highlight=10,
    fontsize=3,
):
    fig, axs = plt.subplots(
        1, len(columns), figsize=(2 * len(columns), 2), sharex="all", sharey="all"
    )

    for i, s in enumerate(columns):
        ax = axs[i]

        df_s = pd.concat([
            df[f"{xprefix}_{s}"].rename("x"),
            df[f"{yprefix}_{s}"].rename("y"),
        ], axis=1, sort=False)

        if density:
            df_s["z"] = CrispyPlot.density_interpolate(df_s["x"], df_s["y"])
            df_s = df_s.sort_values("z")

        ax.scatter(df_s["x"], df_s["y"], c=df_s["z"] if density else CrispyPlot.PAL_DBGD[0], marker="o", edgecolor="", cmap="Spectral_r" if density else None, s=3)

        ax.axhline(0, ls=":", lw=0.1, c="#484848", zorder=0)
        ax.axvline(0, ls=":", lw=0.1, c="#484848", zorder=0)

        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, ls=":", lw=0.1, c="#484848", zorder=0)

        if highlight_text is not None:
            diff = (df_s["x"] - df_s["y"]).sort_values()
            idxs = list(diff.head(n_highlight).index) + list(diff.tail(n_highlight).index)

            texts = [
                ax.text(
                    df_s["x"].loc[i],
                    df_s["y"].loc[i],
                    ";".join(df.loc[i, highlight_text]),
                    color="k",
                    fontsize=fontsize,
                )
                for i in idxs
            ]
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
                ax=ax,
            )

        if add_corr:
            r, p = spearmanr(df_s["x"], df_s["y"])
            ax.annotate(
                f"R={r:.2f},p={p:.1e}" if p != 0 else f"R={r:.2f},p<0.0001",
                xy=(0.01, 0.01),
                xycoords=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=5,
            )

        ax.set_title(f"{s} (N={df_s.shape[0]})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if i == 0 else "")

    fig.subplots_adjust(wspace=0.05, hspace=0.05)


if __name__ == '__main__':
    # Samplesheet
    #
    id_info = ["sgRNA1", "gene1", "sgRNA2", "gene2", "scaffold_type", "linker_type"]
    lib = pd.read_excel(f"{DPATH}/DualGuidePilot_v1.1_annotated.xlsx", index_col=0).query("duplicates == 1")
    lib_ids = lib.reset_index().set_index(id_info)["sgRNA_ID"]

    lib_ss = pd.read_excel(f"{DPATH}/run227_samplesheet.xlsx")
    lib_ss = lib_ss[~lib_ss["name"].apply(lambda v: v.endswith("-2"))]

    samples = ["DNA12", "DNA22", "DNA32", "Lib1", "Lib2", "Lib3", "Oligo14", "Oligo26"]
    samples_pal = lib_ss.groupby("name")["palette"].first()

    # Counts
    #
    counts = ReadCounts(pd.read_excel(f"{RPATH}/fastq_samples_counts_run227.xlsx", index_col=0))
    counts = counts.loc[lib.index].replace(np.nan, 0).astype(int)

    # Fold-changes
    #
    fc = counts.norm_rpm().foldchange(dict(DNA12=["Lib1"], DNA22=["Lib2"], DNA32=["Lib3"]))

    # Export
    #
    pd.concat([
        lib,
        counts.add_suffix(" rawcounts"),
        fc.add_suffix(" log2fc"),
    ], axis=1, sort=False).to_excel(f"{RPATH}/fastq_samples_fcs_run227.xlsx")

    # Counts pairgrid
    #
    pairgrid(np.log2(counts.add(1).norm_rpm()))
    plt.suptitle("Counts - log2(Reads per million)", y=1.02)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(f"{RPATH}/comparisions_pairgrid_counts.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    # Fold-change pairgrid
    #
    pairgrid(fc)
    plt.suptitle("Counts - log2 fold-change", y=1.02)
    plt.gcf().set_size_inches(4, 4)
    plt.savefig(f"{RPATH}/comparisions_pairgrid_fc.png", bbox_inches="tight", dpi=600)
    plt.close("all")

    # Q1: Vector Position
    #
    vp_sgrans = lib.query("(duplicates == 1) & (linker_type == 'wildtype')").reset_index()
    vp_sgrans = vp_sgrans[(vp_sgrans["status1"] == "CoreLOF") & vp_sgrans["status2"].isin(["CoreNon", "NonTarg"])]
    vp_sgrans["sgRNA_ID_pair"] = lib_ids.loc[
        [
            (sg2, g2, sg1, g1, s, l)
            for sg1, g1, sg2, g2, s, l in vp_sgrans[id_info].values
        ]
    ].values
    vp_sgrans = {
        (p1, p2) for p1, p2 in vp_sgrans[["sgRNA_ID", "sgRNA_ID_pair"]].dropna().values
    }
    vp_sgrans = pd.concat(
        [
            lib.loc[[sg1 for sg1, _ in vp_sgrans], id_info + ["status1", "status2"]]
            .reset_index()
            .add_prefix("X_"),
            lib.loc[[sg2 for _, sg2 in vp_sgrans], id_info + ["status1", "status2"]]
            .reset_index()
            .add_prefix("Y_"),
        ],
        axis=1,
        sort=False,
    )
    vp_sgrans = pd.concat(
        [
            vp_sgrans,
            fc.loc[vp_sgrans["X_sgRNA_ID"]].add_prefix("X_").set_index(vp_sgrans.index),
            fc.loc[vp_sgrans["Y_sgRNA_ID"]].add_prefix("Y_").set_index(vp_sgrans.index),
        ],
        axis=1,
        sort=False,
    )
    vp_sgrans.to_excel(
        f"{RPATH}/comparisions_vectorposition_matrix.xlsx", index=False
    )

    for st in ["wildtype", "modified", "both"]:
        plot_df = vp_sgrans.query(f"X_scaffold_type == '{st}'") if st != "both" else vp_sgrans.copy()
        scattergrid(
            plot_df,
            columns=["DNA12", "DNA22", "DNA32"],
            xprefix="X",
            yprefix="Y",
            highlight_text=None,
            xlabel="<CoreLOF,->\nlog2 fold-change",
            ylabel="<-,CoreLOF>\nlog2 fold-change",
        )
        plt.suptitle(f"Scaffold type = {st}", y=1.03)
        plt.savefig(
            f"{RPATH}/comparisions_vectorposition_corrplot_{st}.pdf", bbox_inches="tight"
        )
        plt.close("all")

    plot_df_rename = dict(
        X_DNA12="DNA12 <E,->",
        X_DNA22="DNA22 <E,->",
        X_DNA32="DNA32 <E,->",
        Y_DNA12="DNA12 <-,E>",
        Y_DNA22="DNA22 <-,E>",
        Y_DNA32="DNA32 <-,E>",
    )

    plot_df = vp_sgrans[list(plot_df_rename) + ["X_scaffold_type"]].rename(columns=plot_df_rename)
    plot_df = pd.melt(plot_df, id_vars="X_scaffold_type")

    scaffold_pal = dict(wildtype="#1f77b4", modified="#ff7f0e")

    _, ax = plt.subplots(1, 1, figsize=(3, 2), sharex="all", sharey="all")
    sns.boxplot(
        x="value", y="variable",
        hue="X_scaffold_type",
        data=plot_df,
        palette=scaffold_pal,
        orient="h",
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
        ax=ax,
    )
    ax.set_xlabel("Vector log2 fold-change")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")
    ax.set_ylabel("")
    plt.legend(frameon=False)
    plt.savefig(
        f"{RPATH}/comparisions_vectorposition_boxplot.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Negative controls
    #
    nc_sgrans = lib.query("(duplicates == 1) & (linker_type == 'wildtype')").reset_index()
    nc_sgrans = nc_sgrans[nc_sgrans["status1"].isin(["CoreNon", "NonTarg"]) & nc_sgrans["status2"].isin(["CoreNon", "NonTarg"])]
    nc_sgrans["sgRNA_ID_pair"] = lib_ids.loc[
        [
            (sg2, g2, sg1, g1, s, l)
            for sg1, g1, sg2, g2, s, l in nc_sgrans[id_info].values
        ]
    ].values
    nc_sgrans = {
        (p1, p2) for p1, p2 in nc_sgrans[["sgRNA_ID", "sgRNA_ID_pair"]].dropna().values
    }
    nc_sgrans = pd.concat(
        [
            lib.loc[[sg1 for sg1, _ in nc_sgrans], id_info + ["status1", "status2"]]
            .reset_index()
            .add_prefix("X_"),
            lib.loc[[sg2 for _, sg2 in nc_sgrans], id_info + ["status1", "status2"]]
            .reset_index()
            .add_prefix("Y_"),
        ],
        axis=1,
        sort=False,
    )
    nc_sgrans = pd.concat(
        [
            nc_sgrans,
            fc.loc[nc_sgrans["X_sgRNA_ID"]].add_prefix("X_").set_index(nc_sgrans.index),
            fc.loc[nc_sgrans["Y_sgRNA_ID"]].add_prefix("Y_").set_index(nc_sgrans.index),
        ],
        axis=1,
        sort=False,
    )
    nc_sgrans.to_excel(
        f"{RPATH}/comparisions_negativecontrols_matrix.xlsx", index=False
    )

    plot_df_rename = dict(
        X_DNA12="DNA12 <-,->",
        X_DNA22="DNA22 <-,->",
        X_DNA32="DNA32 <-,->",
        Y_DNA12="DNA12 <-,->",
        Y_DNA22="DNA22 <-,->",
        Y_DNA32="DNA32 <-,->",
    )

    plot_df = nc_sgrans[list(plot_df_rename) + ["X_scaffold_type"]].rename(columns=plot_df_rename)
    plot_df = pd.melt(plot_df, id_vars="X_scaffold_type")

    scaffold_pal = dict(wildtype="#1f77b4", modified="#ff7f0e")

    _, ax = plt.subplots(1, 1, figsize=(3, 2), sharex="all", sharey="all")
    sns.boxplot(
        x="value", y="variable",
        hue="X_scaffold_type",
        data=plot_df,
        palette=scaffold_pal,
        orient="h",
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
        ax=ax,
    )
    ax.set_xlabel("Vector log2 fold-change")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")
    ax.set_ylabel("")
    plt.legend(frameon=False)
    plt.savefig(
        f"{RPATH}/comparisions_negativecontrols_boxplot.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Q2: Backbone
    #
    bb_sgrnas = lib.query(
        "(scaffold_type == 'wildtype') & (duplicates == 1) & (linker_type == 'wildtype')"
    ).reset_index()
    bb_sgrnas["sgRNA_ID_pair"] = lib_ids.loc[
        [
            (sg1, g1, sg2, g2, "modified", l)
            for sg1, g1, sg2, g2, s, l in bb_sgrnas[id_info].values
        ]
    ].values
    bb_sgrnas = {
        (p1, p2) for p1, p2 in bb_sgrnas[["sgRNA_ID", "sgRNA_ID_pair"]].dropna().values
    }

    bb_sgrnas = pd.concat(
        [
            lib.loc[[sg1 for sg1, _ in bb_sgrnas], id_info + ["status1", "status2"]]
            .reset_index()
            .add_prefix("WT_"),
            lib.loc[[sg2 for _, sg2 in bb_sgrnas], id_info + ["status1", "status2"]]
            .reset_index()
            .add_prefix("MT_"),
        ],
        axis=1,
        sort=False,
    )
    bb_sgrnas = pd.concat(
        [
            bb_sgrnas,
            fc.loc[bb_sgrnas["WT_sgRNA_ID"]]
            .add_prefix("WT_")
            .set_index(bb_sgrnas.index),
            fc.loc[bb_sgrnas["MT_sgRNA_ID"]]
            .add_prefix("MT_")
            .set_index(bb_sgrnas.index),
        ],
        axis=1,
        sort=False,
    )
    bb_sgrnas.to_excel(f"{RPATH}/comparisions_backbone_matrix.xlsx", index=False)

    scattergrid(
        df=bb_sgrnas,
        columns=["DNA12", "DNA22", "DNA32"],
        xprefix="WT",
        yprefix="MT",
        highlight_text=["WT_gene1", "WT_gene2"],
        xlabel="Scaffold wildtype\nlog2 fold-change",
        ylabel="Scaffold modified\nlog2 fold-change",
        n_highlight=3,
    )
    plt.savefig(
        f"{RPATH}/comparisions_backbone_corrplot.pdf", bbox_inches="tight"
    )
    plt.close("all")

    plot_df_rename = dict(
        WT_DNA12="DNA12 Wildtype",
        WT_DNA22="DNA22 Wildtype",
        WT_DNA32="DNA32 Wildtype",
        MT_DNA12="DNA12 Modified",
        MT_DNA22="DNA22 Modified",
        MT_DNA32="DNA32 Modified",
    )
    order = [
        "DNA12 Wildtype",
        "DNA12 Modified",
        "DNA22 Wildtype",
        "DNA22 Modified",
        "DNA32 Wildtype",
        "DNA32 Modified",
    ]
    color = samples_pal[[i.split(" ")[0] for i in order]]
    plot_df = bb_sgrnas.iloc[:, -6:].rename(columns=plot_df_rename)

    _, ax = plt.subplots(1, 1, figsize=(3, 2), sharex="all", sharey="all")
    sns.boxplot(
        data=plot_df[order],
        orient="h",
        palette=list(color),
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
        ax=ax,
    )
    ax.set_xlabel("Vector log2 fold-change")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.savefig(
        f"{RPATH}/comparisions_backbone_boxplot.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Q3: First base
    #
    fb_sgrnas = lib.query(
        "(scaffold_type == 'modified') & (duplicates == 1) & (linker_type == 'wildtype')"
    ).reset_index()
    fb_sgrnas["sgRNA1_19"] = fb_sgrnas["sgRNA1"].apply(lambda v: v[1:])
    fb_sgrnas["sgRNA2_19"] = fb_sgrnas["sgRNA2"].apply(lambda v: v[1:])

    fb_sgrnas = (
        pd.concat(
            [
                fb_sgrnas.groupby(["sgRNA1_19", "gene1", "sgRNA2", "gene2"])["sgRNA_ID"]
                .count()
                .to_frame()
                .assign(guide="first"),
                fb_sgrnas.groupby(["sgRNA1", "gene1", "sgRNA2_19", "gene2"])["sgRNA_ID"]
                .count()
                .to_frame()
                .assign(guide="second"),
            ]
        )
        .query("sgRNA_ID == 4")
        .drop(columns=["sgRNA_ID"])
    )

    fb_sgrnas = pd.concat(
        [
            fb_sgrnas.reset_index(),
            pd.concat(
                [
                    fc.loc[
                        lib_ids.loc[
                            [
                                (
                                    bp + sg1 if r == "first" else sg1,
                                    g1,
                                    bp + sg2 if r == "second" else sg2,
                                    g2,
                                    "modified",
                                    "wildtype",
                                )
                                for (sg1, g1, sg2, g2), r in fb_sgrnas[
                                    "guide"
                                ].iteritems()
                            ]
                        ]
                    ]
                    .add_suffix(f"_{bp}")
                    .reset_index(drop=True)
                    for bp in ["G", "T", "A", "C"]
                ],
                axis=1,
                sort=False,
            ),
        ],
        axis=1,
        sort=False,
    )

    fb_sgrnas.to_excel(
        f"{RPATH}/comparisions_firstbase_matrix.xlsx", index=False
    )

    fb_pal = dict(first="#fdd0a2", second="#e6550d")
    xy_min, xy_max = (
        fb_sgrnas.iloc[:, -12:].min().min(),
        fb_sgrnas.iloc[:, -12:].max().max(),
    )
    fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharex="all", sharey="all")
    for i, s in enumerate(["DNA12", "DNA22", "DNA32"]):
        for j, bp in enumerate(["A", "T", "C"]):
            ax = axs[i][j]

            for gtype in fb_pal:
                x = fb_sgrnas.query(f"guide == '{gtype}'")[f"{s}_{bp}"]
                y = fb_sgrnas.query(f"guide == '{gtype}'")[f"{s}_G"]

                ax.scatter(
                    x,
                    y,
                    c=fb_pal[gtype],
                    label="sgRNA1" if gtype == "first" else "sgRNA2",
                    s=10,
                    linewidth=0.1,
                    edgecolor="white",
                    alpha=0.75,
                )

            r, p = spearmanr(fb_sgrnas[f"{s}_{bp}"], fb_sgrnas[f"{s}_G"])
            ax.annotate(
                f"R={r:.2f},p={p:.1e}" if p != 0 else f"R={r:.2f},p<0.0001",
                xy=(0.01, 0.01),
                xycoords=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=5,
            )

            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

            ax.set_xlabel(f"{bp} first base\n(log2 FC)" if i == 2 else "")
            ax.set_ylabel(f"G first base - {s}\n(log2 FC)" if j == 0 else "")

            ax.legend(frameon=False, prop=dict(size=4))

            ax.plot(
                (xy_min, xy_max),
                (xy_min, xy_max),
                ls=":",
                lw=0.1,
                c="#484848",
                zorder=0,
            )
            ax.set_xlim(xy_min, xy_max)
            ax.set_ylim(xy_min, xy_max)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(
        f"{RPATH}/comparisions_firstbase_scattergrid.pdf", bbox_inches="tight"
    )
    plt.close("all")

    # Recall essential and non-essential
    #
    gsets = dict(
        ess=Utils.get_essential_genes(return_series=True),
        ness=Utils.get_non_essential_genes(return_series=True),
    )
    sgsets = dict(
        ess_ness=set(
            lib[lib["gene1"].isin(gsets["ess"]) & (lib["status2"] == "CoreNon")].index
        ),
        ess_ntrg=set(
            lib[lib["gene1"].isin(gsets["ess"]) & (lib["status2"] == "NonTarg")].index
        ),
        ness_ess=set(
            lib[(lib["status1"] == "CoreNon") & lib["gene2"].isin(gsets["ess"])].index
        ),
        ntrg_ess=set(
            lib[(lib["status1"] == "NonTarg") & lib["gene2"].isin(gsets["ess"])].index
        ),
        ness_ntrg=set(
            lib[(lib["status1"] == "CoreNon") & (lib["status2"] == "NonTarg")].index
        ),
        ntrg_ness=set(
            lib[(lib["status1"] == "NonTarg") & (lib["status2"] == "CoreNon")].index
        ),
    )
    pal_ess = dict(
        ess_ness="#3182bd",
        ess_ntrg="#c6dbef",
        ness_ess="#31a354",
        ntrg_ess="#c7e9c0",
        ness_ntrg="#636363",
        ntrg_ness="#d9d9d9",
    )
    sgnames = dict(
        ess_ness="<Ess;NonEss>",
        ess_ntrg="<Ess;NonTarg>",
        ness_ess="<NonEss;Ess>",
        ntrg_ess="<NonTarg;Ess>",
        ness_ntrg="<NonEss;NonTarg>",
        ntrg_ness="<NonTarg;NonEss>",
    )

    fig, axs = plt.subplots(2, 3, figsize=(6, 4), sharex="all", sharey="all")
    for i, n in enumerate(gsets):
        for j, s in enumerate(["DNA12", "DNA22", "DNA32"]):
            ax = axs[i][j]

            gs = (
                ["ess_ness", "ess_ntrg", "ness_ess", "ntrg_ess"]
                if i == 0
                else ["ness_ntrg", "ntrg_ness"]
            )

            for g in gs:
                x, y, xy_auc = QCplot.recall_curve(fc[s], sgsets[g])
                ax.plot(
                    x,
                    y,
                    label=f"{sgnames[g]}={xy_auc:.2f}",
                    lw=1.0,
                    c=pal_ess[g],
                    alpha=0.8,
                )

            ax.plot((0, 1), (0, 1), "k--", lw=0.3, alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            ax.set_xlabel("Percent-rank of vectors" if i == 1 else "")
            ax.set_ylabel("Cumulative fraction" if j == 0 else "")
            ax.set_title(s if i == 0 else "")

            ax.legend(frameon=False, prop=dict(size=4), loc=4 if i == 0 else 2)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f"{RPATH}/comparisions_recall.pdf", bbox_inches="tight")
    plt.close("all")

    # Q4: Linker
    #
    l_gpairs = [
        ("RPL13", "IFNW1"),
        ("RPA1", "IFNW1"),
        ("SRCAP", "IFNW1"),
        ("TCERG1", "IFNW1"),
    ]
    l_gpairs = [(g1, g2) for p in l_gpairs for g1, g2 in [p, p[::-1]]]

    l_sgrnas = lib.query(
        "(scaffold_type == 'modified') & (duplicates == 1) & (sgRNA1_first == 'G') & (sgRNA2_first == 'G')"
    )
    l_sgrnas = l_sgrnas[
        [(g1, g2) in l_gpairs for g1, g2 in l_sgrnas[["gene1", "gene2"]].values]
    ]
    l_sgrnas = pd.concat([l_sgrnas, fc.loc[l_sgrnas.index]], axis=1, sort=False)

    plot_df = l_sgrnas[["DNA12", "DNA22", "DNA32", "linker_type"]]
    plot_df = pd.melt(plot_df, id_vars="linker_type")
    plot_df["name"] = [
        f"{i} - {v}"
        for i, v in lib.groupby("linker_type")["linker"]
        .first()
        .loc[plot_df["linker_type"]]
        .iteritems()
    ]

    order = natsorted(set(plot_df["name"]))

    _, ax = plt.subplots(1, 1, figsize=(2, 5), sharex="all", sharey="all")
    sns.boxplot(
        "value",
        "name",
        "variable",
        data=plot_df,
        order=order,
        orient="h",
        palette=samples_pal,
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
        ax=ax,
    )
    ax.set_xlabel("Vector log2 fold-change")
    ax.set_ylabel("")
    ax.set_title("Linker")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
    ax.legend(frameon=False, prop=dict(size=6), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{RPATH}/comparisions_linkers_boxplot.pdf", bbox_inches="tight")
    plt.close("all")

    # Gene-interactions
    #
    gi_pairs = pd.Series({i: ";".join(v) for i, v in lib[["gene1", "gene2"]].iterrows()})
    gi_fcs = fc.groupby(gi_pairs).mean()

    fig, axs = plt.subplots(1, 3, figsize=(9, 2), sharex="all", sharey="all")

    for i, s in enumerate(["DNA12", "DNA22", "DNA32"]):
        ax = axs[i]

        plot_df = gi_fcs[s].sort_values().reset_index()

        ax.scatter(plot_df.index, plot_df[s], c=samples_pal[s], s=5, linewidths=0)

        ax.axhline(0, ls="-", color="k", lw=.3, zorder=0)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        texts = [
            ax.text(
                i,
                r[s],
                r["index"],
                color="k",
                fontsize=3,
            )
            for i, r in plot_df.head(20).append(plot_df.tail(5)).iterrows()
        ]
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
            ax=ax,
        )

        ax.set_xlabel("Rank of gene pairs")
        ax.set_ylabel("Gene pair log2 fold-change" if i == 0 else "")

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f"{RPATH}/comparisions_geneinteractions_waterfall.pdf", bbox_inches="tight")
    plt.close("all")
