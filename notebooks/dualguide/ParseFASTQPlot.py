import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import logger as LOG
from crispy.CrispyPlot import CrispyPlot
from dualguide import read_gi_library
from dualguide.ParseFASTQ import SCAFFOLDS_MOTIFS

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
