import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.LibRepresentationReport import LibraryRepresentaion

DPATH = pkg_resources.resource_filename("data", "dualguide/")
RPATH = pkg_resources.resource_filename("notebooks", "dualguide/reports/")

if __name__ == '__main__':
    # Samplesheet
    #
    lib = pd.read_excel(f"{DPATH}/DualGuidePilot_v1.1_annotated.xlsx", index_col=0)

    lib_ss = pd.read_excel(f"{DPATH}/run227_samplesheet.xlsx")
    lib_ss = lib_ss[~lib_ss["name"].apply(lambda v: v.endswith("-2"))]

    samples = ["DNA12", "DNA22", "DNA32", "Lib1", "Lib2", "Lib3", "Oligo14", "Oligo26"]
    samples_pal = lib_ss.groupby("name")["palette"].first()

    # Counts
    #
    counts_df = pd.read_excel(f"{RPATH}/fastq_samples_counts_run227.xlsx", index_col=0)
    counts_df = counts_df.loc[lib.query("duplicates == 1").index].replace(np.nan, 0).astype(int)

    # Library representation reports
    #
    lib_report = LibraryRepresentaion(counts_df)

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

    plt.savefig(f"{RPATH}/librepresentation_gini_barplot.pdf", bbox_inches="tight")
    plt.close("all")

    # Lorenz curves
    #
    lib_report.lorenz_curve(palette=samples_pal)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f"{RPATH}/librepresentation_lorenz_curve.pdf", bbox_inches="tight", dpi=600)
    plt.close("all")

    # sgRNA counts boxplots
    #
    lib_report.boxplot(palette=samples_pal)
    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(f"{RPATH}/librepresentation_counts_boxplots.pdf", bbox_inches="tight", dpi=600)
    plt.close("all")

    # sgRNA counts histograms
    #
    lib_report.distplot(palette=samples_pal)
    plt.gcf().set_size_inches(2.5, 2)
    plt.savefig(f"{RPATH}/librepresentation_counts_histograms.pdf", bbox_inches="tight")
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

    plt.savefig(f"{RPATH}/librepresentation_fc_range_barplot.pdf", bbox_inches="tight")
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

    plt.savefig(f"{RPATH}/librepresentation_dropout_barplot.pdf", bbox_inches="tight")
    plt.close("all")
