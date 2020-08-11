import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from GIPlot import GIPlot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from adjustText import adjust_text
from dualguide import read_gi_library
from crispy.DataImporter import CRISPR
from crispy.CRISPRData import ReadCounts
from crispy import CrispyPlot, QCplot, Utils


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
            idxs = list(diff.head(ntop_annotate).index) + list(
                diff.tail(ntop_annotate).index
            )

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
        r, p = spearmanr(df.iloc[:, i], df.iloc[:, j], nan_policy="omit")
        rmse = sqrt(mean_squared_error(df.iloc[:, i], df.iloc[:, j]))
        ax.annotate(
            f"R={r:.2f}\nRMSE={rmse:.2f}",
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

        df_s = pd.concat(
            [df[f"{xprefix}_{s}"].rename("x"), df[f"{yprefix}_{s}"].rename("y")],
            axis=1,
            sort=False,
        ).dropna(subset=["x", "y"])

        if density:
            df_s["z"] = CrispyPlot.density_interpolate(df_s["x"], df_s["y"])
            df_s = df_s.sort_values("z")

        ax.scatter(
            df_s["x"],
            df_s["y"],
            c=df_s["z"] if density else CrispyPlot.PAL_DBGD[0],
            marker="o",
            edgecolor="",
            cmap="Spectral_r" if density else None,
            s=3,
        )

        ax.axhline(0, ls=":", lw=0.1, c="#484848", zorder=0)
        ax.axvline(0, ls=":", lw=0.1, c="#484848", zorder=0)

        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, ls=":", lw=0.1, c="#484848", zorder=0)

        if highlight_text is not None:
            diff = (df_s["x"] - df_s["y"]).sort_values()
            idxs = list(diff.head(n_highlight).index) + list(
                diff.tail(n_highlight).index
            )

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


ESS_GENES = Utils.get_essential_genes(return_series=False)
NESS_GENES = Utils.get_non_essential_genes(return_series=False)


def classify_gene(gene="ARID1A", gene_chr="1"):
    if (str(gene) == "nan") & (str(gene_chr) == "nan"):
        return "non-targeting"

    if (str(gene) == "nan") & (str(gene_chr) != "nan"):
        return "intergenic"

    elif gene in ESS_GENES:
        return "essential"

    elif gene in NESS_GENES:
        return "non-essential"

    else:
        return "unclassified"


if __name__ == "__main__":
    # Project score
    #
    cscore_obj = CRISPR()
    cscore = cscore_obj.filter(dtype="merged")
    cscore_ht29 = cscore["SIDM00136"]

    # Samplesheet
    #
    lib_name = "2gCRISPR_Pilot_library_v2.0.0.xlsx"

    lib_ss = pd.read_excel(f"{DPATH}/gi_samplesheet.xlsx")
    lib_ss = lib_ss.query(f"library == '{lib_name}'")

    lib = read_gi_library(lib_name)
    lib["sgRNA1_class"] = [classify_gene(g, c) for g, c in lib[["sgRNA1_Approved_Symbol", "sgRNA1_Chr"]].values]
    lib["sgRNA2_class"] = [classify_gene(g, c) for g, c in lib[["sgRNA2_Approved_Symbol", "sgRNA2_Chr"]].values]
    lib["vector_class"] = lib["sgRNA1_class"] + " + " + lib["sgRNA2_class"]

    samples = list(set(lib_ss["name"]))
    samples_pal = lib_ss.groupby("name")["palette"].first()

    # Counts
    #
    counts = ReadCounts(
        pd.read_excel(f"{DPATH}/{lib_name}_samples_counts.xlsx", index_col=0)
    )

    # Fold-changes
    #
    comparisons = pd.Series(
        dict(
            GI_PILOT_LIB2_500x_D14_Rep1=["GI_PILOT_LIB2_500x_D3"],
            GI_PILOT_LIB2_500x_D14_Rep2=["GI_PILOT_LIB2_500x_D3"],
            GI_PILOT_LIB2_500x_D14_Rep3=["GI_PILOT_LIB2_500x_D3"],
            GI_PILOT_LIB2_100x_D14_Rep1=["GI_PILOT_LIB2_100x_D3"],
            GI_PILOT_LIB2_100x_D14_Rep2=["GI_PILOT_LIB2_100x_D3"],
            GI_PILOT_LIB2_100x_D14_Rep3=["GI_PILOT_LIB2_100x_D3"],
            GI_PILOT_LIB2_PCR500x_D14_Rep1=["GI_PILOT_LIB2_PCR500x_D3"],
            GI_PILOT_LIB2_PCR500x_D14_Rep2=["GI_PILOT_LIB2_PCR500x_D3"],
            GI_PILOT_LIB2_PCR500x_D14_Rep3=["GI_PILOT_LIB2_PCR500x_D3"],
        )
    )

    comparisons_plasmid = pd.Series(
        dict(
            GI_PILOT_LIB2_500x_D14_Rep1 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_500x_D14_Rep2 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_500x_D14_Rep3 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_100x_D14_Rep1 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_100x_D14_Rep2 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_100x_D14_Rep3 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_PCR500x_D14_Rep1 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_PCR500x_D14_Rep2 = ["GI_PILOT_LIB2_500x_Plasmid"],
            GI_PILOT_LIB2_PCR500x_D14_Rep3 = ["GI_PILOT_LIB2_500x_Plasmid"],
        )
    )

    fc = counts.norm_rpm().foldchange(comparisons.to_dict())
    fc_plasmid = counts.norm_rpm().foldchange(comparisons_plasmid.to_dict())

    # Export
    #
    data = pd.concat([lib, counts.add_suffix(" rawcounts"), fc], axis=1, sort=False)
    data_plasmid = pd.concat([lib, counts.add_suffix(" rawcounts"), fc_plasmid], axis=1, sort=False)

    data["FC_500x"] = data[[f"GI_PILOT_LIB2_500x_D14_Rep{d+1}" for d in range(3)]].mean(1)
    data["FC_500x_Plasmid"] = fc_plasmid[[f"GI_PILOT_LIB2_500x_D14_Rep{d+1}" for d in range(3)]].mean(1)

    data["FC_100x"] = data[[f"GI_PILOT_LIB2_100x_D14_Rep{d+1}" for d in range(3)]].mean(1)
    data["FC_100x_Plasmid"] = fc_plasmid[[f"GI_PILOT_LIB2_100x_D14_Rep{d+1}" for d in range(3)]].mean(1)

    data["FC_PCR500x"] = data[[f"GI_PILOT_LIB2_PCR500x_D14_Rep{d+1}" for d in range(3)]].mean(1)
    data["FC_PCR500x_Plasmid"] = fc_plasmid[[f"GI_PILOT_LIB2_PCR500x_D14_Rep{d+1}" for d in range(3)]].mean(1)

    data.to_excel(f"{RPATH}/{lib_name}_data.xlsx")

    # Counts pairgrid
    #
    pairgrid(np.log2(counts.add(1).norm_rpm()))
    plt.suptitle("Counts (log2 Reads per million)")
    plt.gcf().set_size_inches(12, 12)
    plt.savefig(
        f"{RPATH}/{lib_name}_comparisions_pairgrid_counts.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")

    # Fold-change pairgrid
    #
    plot_df = fc.copy()
    plot_df.columns = [c.replace("GI_PILOT_LIB2_", "") for c in plot_df]

    pairgrid(plot_df)
    plt.suptitle("Log2 fold-change")
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(
        f"{RPATH}/{lib_name}_comparisions_pairgrid_fc.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # Fold-change pairgrid plasmid
    #
    plot_df = fc_plasmid.copy().dropna()
    plot_df.columns = [c.replace("GI_PILOT_LIB2_", "") for c in plot_df]

    pairgrid(plot_df)
    plt.suptitle("Log2 fold-change - Plasmid")
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(
        f"{RPATH}/{lib_name}_comparisions_pairgrid_fc_plasmid.png", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # Distribution
    #
    plot_df = data[list(comparisons.index) + ["Notes"]]
    plot_df = pd.melt(plot_df, value_vars=comparisons.index, id_vars=["Notes"])

    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.boxplot(
        x="value",
        y="variable",
        hue="Notes",
        data=plot_df,
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
    ax.set_ylabel("")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(
        f"{RPATH}/{lib_name}_comparisions_boxplots.pdf", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # Distribution
    #
    plot_df = data_plasmid[list(comparisons_plasmid.index) + ["Notes"]]
    plot_df = pd.melt(plot_df, value_vars=comparisons_plasmid.index, id_vars=["Notes"])

    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.boxplot(
        x="value",
        y="variable",
        hue="Notes",
        data=plot_df,
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
    ax.set_xlabel("Vector log2 fold-change - Plasmid")
    ax.set_ylabel("")
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(
        f"{RPATH}/{lib_name}_comparisions_boxplots_plasmid.pdf", bbox_inches="tight", dpi=600
    )
    plt.close("all")

    # Distribution by scaffold
    #
    plot_df = data[["Notes", "Scaffold", "FC_500x", "FC_100x", "FC_PCR500x"]]
    plot_df = pd.melt(plot_df, id_vars=["Notes", "Scaffold"], value_vars=["FC_500x", "FC_100x", "FC_PCR500x"])

    g = sns.catplot(
        x="value",
        y="Scaffold",
        hue="Notes",
        col="variable",
        data=plot_df,
        order=["Wildtype", "Modified6", "Modified7"],
        facet_kws=dict(despine=False),
        sharex="row",
        sharey="col",
        height=2.5,
        kind="box",
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
    )
    g.set_axis_labels("Vector log2 fold-change", "")

    for ax in g.axes[0]:
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

    plt.savefig(
        f"{RPATH}/{lib_name}_comparisions_boxplots_scaffold.pdf",
        bbox_inches="tight",
        dpi=600,
    )
    plt.close("all")

    # Scaffold correlation
    #
    plot_df = pd.pivot_table(
        data,
        index=["sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence"],
        columns="Scaffold",
        values=["FC_500x", "FC_100x", "FC_PCR500x"],
    ).dropna()
    plot_df.columns = [f"{c[1]}\n{c[0]}" for c in plot_df.columns]
    plot_df = plot_df[[
        "Wildtype\nFC_500x", "Wildtype\nFC_100x", "Wildtype\nFC_PCR500x",
        "Modified6\nFC_500x", "Modified6\nFC_100x", "Modified6\nFC_PCR500x",
        "Modified7\nFC_500x", "Modified7\nFC_100x", "Modified7\nFC_PCR500x",
    ]]

    pairgrid(plot_df)
    plt.gcf().set_size_inches(8, 8)
    plt.savefig(
        f"{RPATH}/{lib_name}_scaffold_fc_pairgrid.png", bbox_inches="tight", dpi=600,
    )
    plt.close("all")

    # Q1: Vector Position
    #
    data_vp = data[
        ["Notes", "Scaffold", "sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence", "FC_500x", "FC_100x", "FC_PCR500x"]
    ].set_index(["Scaffold", "sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence", "Notes"])

    data_vp_x = data_vp.add_prefix("x_")
    data_vp_y = data_vp.reindex(
        [(s, sg2, sg1, n) for s, sg1, sg2, n in data_vp.index]
    ).add_prefix("y_").set_index(data_vp.index)

    data_vp = pd.concat([data_vp_x, data_vp_y], axis=1).dropna().reset_index()

    for stype, stype_df in data_vp.groupby("Scaffold"):
        for c in ["FC_500x", "FC_100x", "FC_PCR500x"]:
            # sgRNA
            grid = GIPlot.gi_regression(f"x_{c}", f"y_{c}", stype_df)
            grid.set_axis_labels(
                "<sgRNA1, sgRNA2>\nlog2 fold-change", "<sgRNA2, sgRNA1>\nlog2 fold-change"
            )
            grid.ax_marg_x.set_title(f"{c} - {stype}")
            plt.savefig(
                f"{RPATH}/{lib_name}_vectorposition_corrplot_{stype}_{c}.pdf",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close("all")

    # Recall essential and non-essential
    #

    vector_status_pal = [
        ("essential + non-targeting", "#ff7f0e"),
        ("non-targeting + essential", "#ffbb78"),

        ("intergenic + essential", "#2ca02c"),
        ("essential + intergenic", "#98df8a"),

        ("non-essential + non-essential", "#3182bd"),
        ("non-targeting + non-targeting", "#393b79"),
        ("intergenic + intergenic", "#756bb1"),
    ]

    for c in ["FC_500x", "FC_500x_Plasmid", "FC_100x", "FC_100x_Plasmid", "FC_PCR500x", "FC_PCR500x_Plasmid"]:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))

        for stype, stype_color in vector_status_pal:
            stype_df = data[~data["Notes"].isin(["DistanceCut"])]
            stype_index_set = set(stype_df.query(f"vector_class == '{stype}'").index)

            x, y, xy_auc = QCplot.recall_curve(stype_df[c], stype_index_set)
            ax.plot(
                x,
                y,
                label=f"{stype} = {xy_auc:.2f} (N={len(stype_index_set)})",
                lw=1.0,
                c=stype_color,
                alpha=0.8,
            )

        ax.plot((0, 1), (0, 1), "k--", lw=0.3, alpha=0.5)

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xlabel("Percent-rank of vectors")
        ax.set_ylabel("Cumulative fraction")

        ax.set_title(c)

        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.savefig(f"{RPATH}/{lib_name}_recall_curves_{c}.pdf", bbox_inches="tight")
        plt.close("all")

        # Recall essential and non-essential per scaffold
        #

        shared_vectors = set.intersection(
            {(sg1, sg2) for sg1, sg2 in data.query(f"Scaffold == 'Wildtype'")[["sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence"]].values},
            {(sg1, sg2) for sg1, sg2 in data.query(f"Scaffold == 'Modified6'")[["sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence"]].values},
            {(sg1, sg2) for sg1, sg2 in data.query(f"Scaffold == 'Modified7'")[["sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence"]].values},
        )

        for c in ["FC_500x", "FC_100x", "FC_PCR500x"]:
            fig, axs = plt.subplots(3, 1, figsize=(2, 6), sharex="all", sharey="all")

            for rowi, stype in enumerate(["Wildtype", "Modified6", "Modified7"]):
                stype_df = data.query(f"Scaffold == '{stype}'")
                stype_df = stype_df[~stype_df["Notes"].isin(["DistanceCut"])]
                stype_df = stype_df[[(sg1, sg2) in shared_vectors for sg1, sg2 in stype_df[["sgRNA1_WGE_Sequence", "sgRNA2_WGE_Sequence"]].values]]

                ax = axs[rowi]

                for vtype, vtype_color in vector_status_pal:
                    vtype_index_set = set(stype_df.query(f"vector_class == '{vtype}'").index)

                    x, y, xy_auc = QCplot.recall_curve(stype_df[c], vtype_index_set)
                    ax.plot(
                        x,
                        y,
                        label=f"{vtype} = {xy_auc:.2f} (N={len(vtype_index_set)})",
                        lw=1.0,
                        c=vtype_color,
                        alpha=0.8,
                    )

                ax.plot((0, 1), (0, 1), "k--", lw=0.3, alpha=0.5)

                ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                ax.set_xlabel("Percent-rank of vectors" if rowi == 2 else "")
                ax.set_ylabel(f"{stype}\nCumulative fraction")

                ax.set_title(c if rowi == 0 else "")

                ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(f"{RPATH}/{lib_name}_recall_curves_{c}_scaffold.pdf", bbox_inches="tight")
            plt.close("all")

        # Single KO comparison
        #
        plot_df = data[data["sgRNA1_class"].isin(["intergenic", "non-targeting"]) | data["sgRNA2_class"].isin(["intergenic", "non-targeting"])]
        plot_df = plot_df[plot_df["sgRNA1_Approved_Symbol"].isin(cscore_ht29.index) | plot_df["sgRNA2_Approved_Symbol"].isin(cscore_ht29.index)]

        plot_df["Gene"] = [gn2 if g1 in ["intergenic", "non-targeting"] else gn1 for g1, g2, gn1, gn2 in plot_df[["sgRNA1_class", "sgRNA2_class", "sgRNA1_Approved_Symbol", "sgRNA2_Approved_Symbol"]].values]
        plot_df["SingleKO"] = [cscore_ht29.loc[g1] if g1 in cscore_ht29.index else cscore_ht29.loc[g2] for g1, g2 in plot_df[["sgRNA1_Approved_Symbol", "sgRNA2_Approved_Symbol"]].values]
        plot_df["ControlType"] = [g1 if g1 in ["intergenic", "non-targeting"] else g2 for g1, g2 in plot_df[["sgRNA1_class", "sgRNA2_class"]].values]

        plot_df = plot_df[["Gene", "ControlType", "SingleKO", "FC_500x", "FC_100x", "FC_PCR500x"]]
        plot_df = plot_df.groupby(["Gene", "ControlType"]).mean().reset_index()

        _, axs = plt.subplots(2, 3, figsize=(6, 4), sharey="all", sharex="all")

        for irow, (ctype, ctype_df) in enumerate(plot_df.groupby("ControlType")):
            for icol, ftype in enumerate(["FC_500x", "FC_100x", "FC_PCR500x"]):
                ax = axs[irow][icol]

                GIPlot.gi_regression_no_marginals(ftype, "SingleKO", plot_df=ctype_df, ax=ax)

                ax.set_title(ftype if irow == 0 else "")
                ax.set_ylabel(f"Control={ctype}\nSingle KO log2 fold-change" if icol == 0 else "")
                ax.set_xlabel(f"Dual KO\nlog2 fold-change" if irow == 1 else "")

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(f"{RPATH}/{lib_name}_singleko_scatter.pdf", bbox_inches="tight")
        plt.close("all")
