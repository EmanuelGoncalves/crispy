import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import logger as LOG
from dualguide import read_gi_library
from crispy.CrispyPlot import CrispyPlot
from dualguide.ParseFASTQ import SCAFFOLDS_MOTIFS
from crispy.LibRepresentationReport import LibraryRepresentaion


DPATH = pkg_resources.resource_filename("data", "dualguide/")
RPATH = pkg_resources.resource_filename("notebooks", "dualguide/reports/")


def parsed_reads_reports(sample):
    c_reads = pd.read_csv(
        f"{DPATH}/gi_counts/{sample}_parsed_reads.csv.gz", index_col=0
    )
    LOG.info(f"Importing {sample} reads")

    # Overall metrics
    c_dict = dict()

    c_dict["total_reads"] = c_reads.shape[0]

    c_dict["sgRNA1_exists_inlib"] = sum(c_reads["sgRNA1_exists_inlib"])
    c_dict["sgRNA2_exists_inlib"] = sum(c_reads["sgRNA2_exists_inlib"])

    c_dict["sgRNA_has_ID"] = sum(1 - c_reads["id"].isna())
    c_dict["sgRNA_pair_exists"] = sum(c_reads["sgRNA_pair_exists"])

    c_dict["swap"] = sum(c_reads["swap"])

    if "_PILOT_" in sample:
        c_dict["scaffold_exists"] = c_reads["Scaffold_Sequence_exists"].sum()

        c_dict["scaffold_WT"] = sum(c_reads["scaffold"] == "Wildtype")
        c_dict["scaffold_MT6"] = sum(c_reads["scaffold"] == "Modified6")
        c_dict["scaffold_MT7"] = sum(c_reads["scaffold"] == "Modified7")

        for s in ["Wildtype", "Modified6", "Modified7"]:
            c_dict[f"swap {s}"] = sum(c_reads.query(f"scaffold == '{s}'")["swap"])

    # Counts
    c_counts = c_reads["id"].dropna().value_counts()

    return dict(stats=c_dict, counts=c_counts)


if __name__ == "__main__":
    # Samplesheet
    #
    lib_ss = pd.read_excel(f"{DPATH}/gi_samplesheet.xlsx")
    lib_ss = lib_ss.query("library == '2gCRISPR_Pilot_library_v2.0.0.xlsx'")

    # lib_name, lib_ss_df = list(lib_ss.groupby("library"))[1]
    for lib_name, lib_ss_df in lib_ss.groupby("library"):
        LOG.info(f"{lib_name}")

        lib = read_gi_library(lib_name)

        samples = list(set(lib_ss_df["name"]))
        samples_pal = lib_ss_df.groupby("name")["palette"].first()

        # FASTQ reads stats, swaps and counts
        #
        s_stats = {s: parsed_reads_reports(s) for s in samples}

        s_reports = pd.DataFrame({s: s_stats[s]["stats"] for s in samples})
        s_reports.to_excel(f"{DPATH}/{lib_name}_samples_stats.xlsx")

        s_counts = pd.DataFrame({s: s_stats[s]["counts"] for s in samples})
        s_counts.to_excel(f"{DPATH}/{lib_name}_samples_counts.xlsx")

        # Total reads
        #
        _, ax = plt.subplots(1, 1, figsize=(0.125 * len(samples), 1.5))
        ax.bar(
            np.arange(len(samples)),
            s_reports.loc["total_reads", samples],
            color=samples_pal.loc[samples],
            lw=0,
        )
        ax.set_xticks(np.arange(len(samples)))
        ax.set_xticklabels(samples, rotation=90)
        ax.set_title("FASTQ reads")
        ax.set_ylabel("Number of reads")
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
        plt.savefig(f"{RPATH}/{lib_name}_total_counts.pdf", bbox_inches="tight")
        plt.close("all")

        # Reports barplots
        #
        plot_df = (s_reports / s_reports.loc["total_reads"]) * 100
        plot_df = pd.melt(plot_df.drop("total_reads").reset_index(), id_vars="index")

        col_order = [
            "sgRNA1_exists_inlib",
            "sgRNA2_exists_inlib",
            "sgRNA_has_ID",
            "sgRNA_pair_exists",
            "swap",
        ]

        if "_PILOT_" in lib_name:
            col_order += [
                "scaffold_exists",
                "scaffold_WT",
                "scaffold_MT6",
                "scaffold_MT7",
                "swap Wildtype",
                "swap Modified6",
                "swap Modified7",
            ]

        f, axs = plt.subplots(
            1,
            len(col_order),
            sharex="all",
            sharey="all",
            figsize=(len(col_order) * len(samples) * 0.3, 3.0),
            dpi=600,
        )

        for i, c in enumerate(col_order):
            ax = axs[i]

            df = plot_df.query(f"index == '{c}'")
            df = df.set_index("variable").loc[samples]

            ax.bar(
                np.arange(len(samples)), df["value"], color=samples_pal[df.index], lw=0
            )

            ax.set_ylim(0, 110)
            ax.set_yticks(np.arange(0, 110, 10))
            ax.set_xticks(np.arange(len(samples)))
            ax.set_xticklabels(samples, rotation=90)

            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

            for idx, o in enumerate(samples):
                ax.text(
                    idx,
                    df.loc[o, "value"],
                    f"{df.loc[o, 'value']:.1f}%",
                    fontsize=6,
                    c="k",
                    rotation=90,
                    ha="center",
                    va="bottom",
                )

            ax.set_title(c, rotation=90)
            ax.set_ylabel("% of total FASTQ reads" if i == 0 else "")
            ax.set_xlabel("")

        plt.subplots_adjust(hspace=0, wspace=0.05)
        plt.savefig(f"{RPATH}/{lib_name}_match_report.pdf", bbox_inches="tight")
        plt.close("all")

        # Library representation
        lib_report = LibraryRepresentaion(s_counts)

        # Comparison gini scores
        #
        gini_scores_comparison = dict(
            avana=0.361, brunello=0.291, yusa_v1=0.342, yusa_v11=0.229
        )

        gini_scores_comparison_palette = dict(
            avana="#66c2a5", brunello="#fc8d62", yusa_v1="#8da0cb", yusa_v11="#e78ac3"
        )

        gini_scores = (
            lib_report.gini().reset_index().rename(columns={"index": "sample", 0: "gini"})
        )

        # Gini scores barplot
        #
        plt.figure(figsize=(2.5, 1.0), dpi=600)

        sns.barplot(
            "gini",
            "sample",
            data=gini_scores,
            orient="h",
            order=samples,
            palette=samples_pal,
            saturation=1,
            lw=0,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        for k, v in gini_scores_comparison.items():
            plt.axvline(
                v, lw=0.5, zorder=1, color=gini_scores_comparison_palette[k], label=k
            )

        plt.xlabel("Gini score")
        plt.ylabel("")

        plt.legend(
            frameon=False, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 4}
        )

        plt.savefig(f"{RPATH}/{lib_name}_librepresentation_gini_barplot.pdf", bbox_inches="tight")
        plt.close("all")

        # Lorenz curves
        #
        lib_report.lorenz_curve(palette=samples_pal)
        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f"{RPATH}/{lib_name}_librepresentation_lorenz_curve.pdf", bbox_inches="tight", dpi=600)
        plt.close("all")

        # sgRNA counts boxplots
        #
        lib_report.boxplot(palette=samples_pal, order=samples_pal.index)
        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f"{RPATH}/{lib_name}_librepresentation_counts_boxplots.pdf", bbox_inches="tight", dpi=600)
        plt.close("all")

        # sgRNA counts histograms
        #
        lib_report.distplot(palette=samples_pal)
        plt.gcf().set_size_inches(2.5, 2)
        plt.savefig(f"{RPATH}/{lib_name}_librepresentation_counts_histograms.pdf", bbox_inches="tight")
        plt.close("all")

        # Percentile scores barplot (good library will have a value below 6)
        #
        percentile_scores = (
            lib_report.percentile()
                .reset_index()
                .rename(columns={"index": "sample", 0: "range"})
        )

        plt.figure(figsize=(2.5, 1.0), dpi=600)

        sns.barplot(
            "range",
            "sample",
            data=percentile_scores,
            orient="h",
            order=samples,
            palette=samples_pal,
            saturation=1,
            lw=0,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.axvline(6, lw=0.5, zorder=1, color="k")

        plt.xlabel("Fold-change range containing 95% of the guides")
        plt.ylabel("")

        plt.savefig(f"{RPATH}/{lib_name}_librepresentation_fc_range_barplot.pdf", bbox_inches="tight")
        plt.close("all")

        # Drop-out rates barplot
        #
        dropout_rates = lib_report.dropout_rate() * 100
        dropout_rates = dropout_rates.reset_index().rename(
            columns={"index": "sample", 0: "dropout"}
        )

        plt.figure(figsize=(2.5, 1.0), dpi=600)

        ax = sns.barplot(
            "dropout",
            "sample",
            data=dropout_rates,
            orient="h",
            order=samples,
            palette=samples_pal,
            saturation=1,
            lw=0,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        vals = ax.get_xticks()
        ax.set_xticklabels([f"{x:.1f}%" for x in vals])

        plt.xlabel("Dropout sgRNAs (zero counts)")
        plt.ylabel("")

        plt.savefig(f"{RPATH}/{lib_name}_librepresentation_dropout_barplot.pdf", bbox_inches="tight")
        plt.close("all")
