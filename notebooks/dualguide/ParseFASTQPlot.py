import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import logger as LOG
from crispy.CrispyPlot import CrispyPlot
from dualguide.ParseFASTQ import SCAFFOLDS_MOTIFS

DPATH = pkg_resources.resource_filename("data", "dualguide/")
RPATH = pkg_resources.resource_filename("notebooks", "dualguide/reports/")


def parsed_reads_reports(sample):
    sample_run = lib_ss.groupby("name")["run"].first()

    c_reads = pd.read_csv(f"{RPATH}/fastq_parsed_run{sample_run[sample]}_{sample}.csv.gz", index_col=0)
    LOG.info(f"Importing {sample} reads")

    # Overall metrics
    c_dict = dict()

    c_dict["total_reads"] = c_reads.shape[0]

    c_dict["scaffold_exists"] = c_reads["scaffold_exists"].sum()
    c_dict["scaffold_WT"] = sum(c_reads["scaffold"] == "wildtype")
    c_dict["scaffold_MT"] = sum(c_reads["scaffold"] == "modified")

    c_dict["linker_exists"] = sum(c_reads["linker_exists"])

    c_dict["sgRNA1_exists_inlib"] = sum(c_reads["sgRNA1_exists_inlib"])
    c_dict["sgRNA2_exists_inlib"] = sum(c_reads["sgRNA2_exists_inlib"])

    c_dict["sgRNAID_exists"] = sum(1 - c_reads["id"].isna())
    c_dict["sgRNA_pair_exists"] = sum(c_reads["sgRNA_pair_exists"])

    c_dict["swap"] = sum(c_reads["swap"])
    for s in ["wildtype", "modified"]:
        c_dict[f"swap {s}"] = sum(c_reads.query(f"scaffold == '{s}'")["swap"])

    for s in ["wildtype", "modified"]:
        c_dict[f"pairs <E,_> {s}"] = c_reads.query(
            f"(status1 == 'CoreLOF') and (status2 != 'CoreLOF') and (scaffold == '{s}')"
        ).shape[0]
        c_dict[f"pairs <_,E> {s}"] = c_reads.query(
            f"(status1 != 'CoreLOF') and (status2 == 'CoreLOF') and (scaffold == '{s}')"
        ).shape[0]
        c_dict[f"pairs <_,_> {s}"] = c_reads.query(
            f"(status1 != 'CoreLOF') and (status2 != 'CoreLOF') and (scaffold == '{s}')"
        ).shape[0]

    # Swap matrix
    c_swap = (
        c_reads.query("swap == 1")
        .assign(count=1)
        .groupby(["sgRNA1", "sgRNA2"])["count"]
        .sum()
        .reset_index()
    )
    c_swap = pd.pivot_table(
        c_swap, index="sgRNA1", columns="sgRNA2", values="count", fill_value=0
    )

    # Counts
    c_counts = c_reads["id"].dropna().value_counts()

    # Backbone
    backbone = pd.concat(
        [c_reads["scaffold"], c_reads["backbone_pos"]], axis=1, sort=False
    )

    # Backbone
    trna = c_reads["trna_pos"]

    # Scaffold
    scaffold = c_reads[SCAFFOLDS_MOTIFS]

    return dict(
        stats=c_dict, swap=c_swap, counts=c_counts, backbone=backbone, trna=trna, scaffold=scaffold
    )


if __name__ == "__main__":
    # Samplesheet
    #
    lib = pd.read_excel(f"{DPATH}/DualGuidePilot_v1.1_annotated.xlsx")

    lib_ss = pd.read_excel(f"{DPATH}/samplesheet.xlsx")

    samples = [
        "DNA12",
        "DNA22",
        "DNA32",
        "Lib1",
        "Lib2",
        "Lib3",
        "Oligo14",
        "Oligo26",
        "DNA12-2",
        "DNA22-2",
        "DNA32-2",
        "Cas9_Day06",
        "Cas9_Day14",
        "Cas9_Day21",
        "Parental_Day06",
        "Parental_Day14",
        "Parental_Day21",
        "Cas9_Day14_PCR18",
        "Parental_Day14_PCR18",
        "Library1",
    ]
    samples_pal = lib_ss.groupby("name")["palette"].first()

    # FASTQ reads stats, swaps and counts
    #
    s_stats = {s: parsed_reads_reports(s) for s in samples}

    s_reports = pd.DataFrame({s: s_stats[s]["stats"] for s in samples})
    s_reports.to_excel(f"{RPATH}/fastq_samples_stats.xlsx")

    s_counts = pd.DataFrame({s: s_stats[s]["counts"] for s in samples})
    s_counts.to_excel(f"{RPATH}/fastq_samples_counts.xlsx")

    b_counts = pd.DataFrame(
        {
            s: s_stats[s]["backbone"]
            .replace(np.nan, "mismatch")
            .reset_index()
            .groupby(["scaffold", "backbone_pos"])["index"]
            .count()
            for s in samples
        }
    )
    b_counts.to_excel(f"{RPATH}/fastq_samples_backbone_counts.xlsx")

    t_counts = pd.DataFrame(
        {
            s: s_stats[s]["trna"].reset_index().groupby("trna_pos")["index"].count()
            for s in samples
        }
    )
    t_counts.to_excel(f"{RPATH}/fastq_samples_trna_counts.xlsx")

    # Scaffold position
    #
    conditions = [
        ("DNA", ["DNA12", "DNA22", "DNA32"]),
        ("Lib", ["Lib1", "Lib2", "Lib3"]),
        ("Oligo", ["Oligo14", "Oligo26"]),
        ("DNA-2", ["DNA12-2", "DNA22-2", "DNA32-2"]),
        ("DayCas9", ["Cas9_Day06", "Cas9_Day14", "Cas9_Day21"]),
        ("ParentalCas9", ["Parental_Day06", "Parental_Day14", "Parental_Day21"]),
        ("Day14PCR18", ["Cas9_Day14_PCR18", "Parental_Day14_PCR18"]),
        ("Library1", ["Library1"]),
    ]

    col_order = ["scaffold_wt", "scaffold_k", "scaffold_mt"]
    position_pal = dict(pos1="#1f77b4", pos2="#ff7f0e")

    for n, ss in conditions:
        fig, axs = plt.subplots(
            len(ss), 3, figsize=(5.5, 1.5 * len(ss)), sharex="all", sharey="all", squeeze=False
        )

        for i, s in enumerate(ss):
            df = pd.DataFrame({c: s_stats[s]["scaffold"][c].value_counts() for c in s_stats[s]["scaffold"]})
            df = df.replace(np.nan, 0)

            for j, stype in enumerate(col_order):
                ax = axs[i][j]
                for pos in position_pal:
                    s_name = f"{stype}_{pos}"
                    ax.bar(
                        df[s_name].index,
                        df[s_name].values,
                        color=position_pal[pos],
                        linewidth=0,
                        alpha=1,
                        label=f"{pos} {SCAFFOLDS_MOTIFS[s_name]}",
                    )
                ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
                ax.set_yscale("log", basey=10)

                if i == 0:
                    ax.set_title(stype)

                if i == (len(ss) - 1):
                    ax.set_xlabel("Scaffold motif\nstart position in R1")

                if j == 0:
                    ax.set_ylabel(f"{s}\nNumber of reads (log10)")

                ax.legend(frameon=False, prop={"size": 4})

        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(f"{RPATH}/scaffold_motifs_histogram_{n}.pdf", bbox_inches="tight")
        plt.close("all")

    scaffold_assembly = pd.Series({
        (1, 1, 0, 0, 0, 0): "WT + WT",
        (0, 0, 1, 1, 0, 0): "K + K",
        (0, 0, 0, 0, 1, 1): "MT + MT",

        (1, 0, 0, 1, 0, 0): "WT + K",
        (1, 0, 1, 0, 0, 0): "WT + K",
        (0, 1, 1, 0, 0, 0): "WT + K",
        (0, 1, 0, 1, 0, 0): "WT + K",

        (1, 0, 0, 0, 0, 1): "WT + MT",
        (0, 1, 0, 0, 0, 1): "WT + MT",
        (0, 1, 0, 0, 1, 0): "WT + MT",
        (1, 0, 0, 0, 1, 0): "WT + MT",

        (0, 0, 1, 0, 1, 0): "MT + K",
        (0, 0, 0, 1, 1, 0): "MT + K",
        (0, 0, 1, 0, 0, 1): "MT + K",
        (0, 0, 0, 1, 0, 1): "MT + K",
    })

    col_order = [
        ("WT", "WT + K", "WT + WT"),
        ("MT", "MT + K", "MT + MT"),
    ]

    ratio_pal = {0: "#1f77b4", 1: "#ff7f0e"}

    for n, ss in conditions:
        fig, axs = plt.subplots(
            len(ss), len(col_order), figsize=(2 * len(col_order), 1.5 * len(ss)), sharex="all", sharey="all", squeeze=False
        )

        for i, s in enumerate(ss):
            df = s_stats[s]["scaffold"][(s_stats[s]["scaffold"] != -1).sum(1) == 2]
            df["stype"] = scaffold_assembly[(df != -1).astype(int).apply(tuple, axis=1)].values
            LOG.info(f"{s}: {df.shape} vs {df.dropna().shape}")

            for j, (stype, numerator, denominator) in enumerate(col_order):
                ax = axs[i][j]

                ratio = []
                for w, stype_hybrid in enumerate([denominator, numerator]):
                    df_stype = df.query(f"stype == '{stype_hybrid}'").drop(columns="stype")
                    ratio.append(df_stype.shape[0])
                    df_stype = df_stype.replace(-1, np.nan).unstack().dropna().value_counts()

                    ax.bar(
                        df_stype.index,
                        df_stype.values,
                        color=ratio_pal[w],
                        linewidth=0,
                        alpha=1,
                        label=stype_hybrid,
                    )
                ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
                ax.set_yscale("log", basey=10)

                if i == 0:
                    ax.set_title(stype)

                if i == (len(ss) - 1):
                    ax.set_xlabel("Scaffold motif\nstart position in R1")

                if j == 0:
                    ax.set_ylabel(f"{s}\nNumber of reads (log10)")

                lgd_title = f"Ratio ({numerator}/{denominator}) = ({ratio[1]}/{ratio[0]}) = {(ratio[1]/ratio[0]*100):.1f}%"
                ax.legend(frameon=False, prop={"size": 4}, title=lgd_title).get_title().set_fontsize("4")

        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(f"{RPATH}/scaffold_hybrid_motifs_histogram_{n}.pdf", bbox_inches="tight")
        plt.close("all")

    # Backbone histogram
    #
    conditions = [
        ("DNA", ["DNA12", "DNA22", "DNA32"]),
        ("Lib", ["Lib1", "Lib2", "Lib3"]),
        ("Oligo", ["Oligo14", "Oligo26"]),
        ("DNA-2", ["DNA12-2", "DNA22-2", "DNA32-2"]),
        ("DayCas9", ["Cas9_Day06", "Cas9_Day14", "Cas9_Day21"]),
        ("ParentalCas9", ["Parental_Day06", "Parental_Day14", "Parental_Day21"]),
        ("Day14PCR18", ["Cas9_Day14_PCR18", "Parental_Day14_PCR18"]),
        ("Library1", ["Library1"]),
    ]

    scaffold_pal = dict(wildtype="#1f77b4", modified="#ff7f0e", mismatch="#76716D")
    col_order = ["modified", "wildtype", "mismatch"]

    for n, ss in conditions:
        fig, axs = plt.subplots(
            len(ss), 3, figsize=(5.5, 1.5 * len(ss)), sharex="col", sharey="col", squeeze=False
        )

        for i, s in enumerate(ss):
            df = s_stats[s]["backbone"].replace(np.NaN, "mismatch")

            for j, stype in enumerate(col_order):
                ax = axs[i][j]
                ax.hist(
                    df.query(f"scaffold == '{stype}'")["backbone_pos"],
                    bins=200,
                    color=scaffold_pal[stype],
                    linewidth=0,
                    alpha=1,
                )
                ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
                ax.set_yscale("log", basey=10)

                if i == 0:
                    ax.set_title(stype)

                if i == (len(ss) - 1):
                    ax.set_xlabel("Backbone motif\n(GTTTAAG) position in R1")

                if j == 0:
                    ax.set_ylabel(f"{s}\nNumber of reads (log10)")

        fig.subplots_adjust(hspace=0.05, wspace=0.25)
        plt.savefig(f"{RPATH}/backbone_position_histogram_{n}.pdf", bbox_inches="tight")
        plt.close("all")

    # Backbone histogram
    #
    for n, ss in conditions:
        fig, axs = plt.subplots(
            len(ss), 1, figsize=(5.5, 1.5 * len(ss)), sharex="col", sharey="col", squeeze=False
        )

        for i, s in enumerate(ss):
            df = s_stats[s]["trna"]

            ax = axs[i][0]
            ax.hist(df, bins=94, color=CrispyPlot.PAL_DBGD[0], linewidth=0, alpha=1)
            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
            ax.set_yscale("log", basey=10)

            if i == (len(ss) - 1):
                ax.set_xlabel("tRNA motif (GCATTGG) position in R2\nreverse complement")

            ax.set_ylabel(f"{s}\nNumber of reads (log10)")

        fig.subplots_adjust(hspace=0.05, wspace=0.25)
        plt.savefig(f"{RPATH}/trna_position_histogram_{n}.pdf", bbox_inches="tight")
        plt.close("all")

    # Total reads
    #
    _, ax = plt.subplots(1, 1, figsize=(.125 * len(samples), 1.5))
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
    plt.savefig(f"{RPATH}/fastq_total_counts.pdf", bbox_inches="tight")
    plt.close("all")

    # Reports barplots
    #
    plot_df = (s_reports / s_reports.loc["total_reads"]) * 100
    plot_df = pd.melt(plot_df.drop("total_reads").reset_index(), id_vars="index")

    col_order = [
        "sgRNA1_exists_inlib",
        "sgRNA2_exists_inlib",
        "scaffold_exists",
        "scaffold_WT",
        "scaffold_MT",
        "linker_exists",
        "sgRNAID_exists",
        "sgRNA_pair_exists",
        "swap",
    ]

    f, axs = plt.subplots(
        1,
        len(col_order),
        sharex="all",
        sharey="all",
        figsize=(len(col_order) * 2.5, 3.0),
        dpi=600,
    )

    for i, c in enumerate(col_order):
        ax = axs[i]

        df = plot_df.query(f"index == '{c}'")
        df = df.set_index("variable").loc[samples]

        ax.bar(np.arange(len(samples)), df["value"], color=samples_pal[df.index], lw=0)

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

        ax.set_title(c)
        ax.set_ylabel("% of total FASTQ reads" if i == 0 else "")
        ax.set_xlabel("")

    plt.subplots_adjust(hspace=0, wspace=0.05)
    plt.savefig(f"{RPATH}/fastq_match_report.pdf", bbox_inches="tight")
    plt.close("all")

    # Pairs barplots
    #
    order = [
        "scaffold_WT",
        "scaffold_MT",
        "pairs <E,_> wildtype",
        "pairs <E,_> modified",
        "pairs <_,E> wildtype",
        "pairs <_,E> modified",
        "pairs <_,_> wildtype",
        "pairs <_,_> modified",
        "swap wildtype",
        "swap modified",
    ]

    plot_df = (s_reports / s_reports.loc["total_reads"]).loc[order]
    plot_df["library"] = [
        lib["scaffold_type"].value_counts()["wildtype"] / lib.shape[0],
        lib["scaffold_type"].value_counts()["modified"] / lib.shape[0],
        lib.query(
            "(status1 == 'CoreLOF') and (status2 != 'CoreLOF') and (scaffold_type == 'wildtype')"
        ).shape[0]
        / lib.shape[0],
        lib.query(
            "(status1 == 'CoreLOF') and (status2 != 'CoreLOF') and (scaffold_type == 'modified')"
        ).shape[0]
        / lib.shape[0],
        lib.query(
            "(status1 != 'CoreLOF') and (status2 == 'CoreLOF') and (scaffold_type == 'wildtype')"
        ).shape[0]
        / lib.shape[0],
        lib.query(
            "(status1 != 'CoreLOF') and (status2 == 'CoreLOF') and (scaffold_type == 'modified')"
        ).shape[0]
        / lib.shape[0],
        lib.query(
            "(status1 != 'CoreLOF') and (status2 != 'CoreLOF') and (scaffold_type == 'wildtype')"
        ).shape[0]
        / lib.shape[0],
        lib.query(
            "(status1 != 'CoreLOF') and (status2 != 'CoreLOF') and (scaffold_type == 'modified')"
        ).shape[0]
        / lib.shape[0],
        0,
        0,
    ]
    plot_df = pd.melt((plot_df * 100).reset_index(), id_vars="index")

    _, ax = plt.subplots(1, 1, figsize=(len(order) * 0.7, 3.0))
    b = sns.barplot(
        "index",
        "value",
        "variable",
        data=plot_df,
        order=order,
        hue_order=samples,
        palette=samples_pal[samples],
        linewidth=0,
        saturation=1,
    )
    b.set_xticklabels(b.get_xticklabels(), rotation=90)
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
    ax.set_xlabel("")
    ax.set_ylabel("% of total FASTQ reads")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 5})
    plt.savefig(f"{RPATH}/fastq_pairs_representation.pdf", bbox_inches="tight")
    plt.close("all")

    # Swap matrix
    #
    order = set(lib["sgRNA1"]).union(lib["sgRNA2"])

    for c in ["DNA", "Lib", "Oligo14", "Oligo26", "Library", "DNA-2"]:
        if c.startswith("Oligo"):
            df = s_stats[c]["swap"].loc[order, order].replace(np.nan, 0).astype(int)

        elif c == "Library":
            df = (
                lib.assign(count=1)
                .groupby(["sgRNA1", "sgRNA2"])["count"]
                .sum()
                .reset_index()
            )
            df = pd.pivot_table(
                df, index="sgRNA1", columns="sgRNA2", values="count", fill_value=0
            )

        else:
            s = f"{(2 if c in ['DNA', 'DNA-2'] else '')}"
            p = f"{('-2' if c == 'DNA-2' else '')}"

            df1, df2, df3 = [
                s_stats[f"{c.split('-')[0]}{i}{s}{p}"]["swap"]
                .loc[order, order]
                .replace(np.nan, 0)
                .astype(int)
                for i in [1, 2, 3]
            ]
            df = df1 + df2 + df3

        _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=600)
        sns.heatmap(
            np.log10(df + 1),
            yticklabels=False,
            xticklabels=False,
            cmap="viridis_r",
            mask=(df == 0),
            ax=ax,
        )
        ax.set_title(c)
        plt.savefig(f"{RPATH}/fastq_swap_heatmap_{c}.png", bbox_inches="tight")
        plt.close("all")
