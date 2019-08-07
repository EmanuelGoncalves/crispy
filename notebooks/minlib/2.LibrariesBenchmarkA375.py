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
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from itertools import combinations
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from minlib import clib_palette, LOG, rpath, dpath


# Kosuke Yusa v1.0

ky_v1_data = CRISPRDataSet("Yusa_v1")

ky_v1_count = ky_v1_data.counts.remove_low_counts(
    ky_v1_data.plasmids, counts_threshold=1
)
ky_v1_count = ky_v1_count[
    [c for c in ky_v1_count if c.startswith("A375")] + ky_v1_data.plasmids
]

ky_v1_library = ky_v1_data.lib.loc[ky_v1_count.index, "Gene"]


# Brunello

brunello_data = CRISPRDataSet("Brunello_A375")

brunello_count = brunello_data.counts.remove_low_counts(
    brunello_data.plasmids, counts_threshold=1
)

brunello_library = brunello_data.lib.loc[brunello_count.index, "Gene"]
brunello_library = brunello_library[
    [not v.startswith("NO_CURRENT_") for v in brunello_library]
]


# GeCKO v2

gecko_data = CRISPRDataSet("GeCKOv2", exclude_guides={})

gecko_count = gecko_data.counts.remove_low_counts(
    gecko_data.plasmids, counts_threshold=1
)
gecko_count = gecko_count[
    [c for c in gecko_count if c.startswith("A375")] + gecko_data.plasmids
]

gecko_library = gecko_data.lib[gecko_data.lib.index.isin(gecko_count.index)]["Gene"]


#

gess = set(pd.read_csv(f"{dpath}/gene_sets/hart_msb_2014_essential.txt", header=None)[0])
gness = set(pd.read_csv(f"{dpath}/gene_sets/hart_msb_2014_nonessential.txt", header=None)[0])


# Assemble CRISPR data-sets

datasets = [
    dict(
        name="GeCKOv2",
        description="v2",
        dataset=gecko_count,
        library=gecko_library,
        plasmids=gecko_data.plasmids,
    ),
    dict(
        name="Yusa_v1",
        description="Day 14",
        dataset=ky_v1_count[list(ky_v1_count.columns[:3]) + ky_v1_data.plasmids],
        library=ky_v1_library,
        plasmids=ky_v1_data.plasmids,
    ),
    dict(
        name="Yusa_v1",
        description="Score",
        dataset=ky_v1_count[list(ky_v1_count.columns[-4:-1]) + ky_v1_data.plasmids],
        library=ky_v1_library,
        plasmids=ky_v1_data.plasmids,
    ),
    dict(
        name="Brunello",
        description="Original tracrRNA",
        dataset=brunello_count[brunello_count.columns[:3]],
        library=brunello_library,
        plasmids=brunello_data.plasmids,
    ),
    dict(
        name="Brunello",
        description="Modified tracrRNA",
        dataset=brunello_count[brunello_count.columns[3:]],
        library=brunello_library,
        plasmids=brunello_data.plasmids,
    ),
]


# Benchmark subsample of library

arocs = []
n_iterations = 10

for dset in datasets:
    guides = dset["library"]

    nguides_thres = int(guides.value_counts().median())

    # Iterate number of guides
    for nguides in np.arange(1, nguides_thres + 1):

        # Bootstrap
        for i in range(n_iterations):
            LOG.info(
                f"{dset['name']} - {dset['description']} - #(guides) = {nguides} : iteration {i}"
            )

            # Guides
            guides_rand = pd.concat(
                [
                    df.sample(n=nguides) if df.shape[0] > nguides else df
                    for n, df in guides.groupby(guides)
                ]
            )

            nguides_count = dset["dataset"].loc[guides_rand.index]
            nguides_fc = nguides_count.norm_rpm().foldchange(dset["plasmids"])

            # Samples
            nsamples = nguides_fc.shape[1]
            samples = [
                list(s)
                for n in np.arange(1, nsamples + 1)
                for s in combinations(nguides_fc, n)
            ]

            # Essential/Non-essential benchmark
            for ss in samples:
                ss_fc = nguides_fc[ss].groupby(guides_rand).mean().mean(1)

                arocs.append(
                    dict(
                        library=dset["name"],
                        description=dset["description"],
                        samples=";".join(ss),
                        n_samples=len(ss),
                        nguides=nguides,
                        aroc=QCplot.pr_curve(ss_fc, true_set=gess, false_set=gness),
                        aupr=QCplot.recall_curve(ss_fc, index_set=gess)[2],
                        fpr=QCplot.aroc_threshold(ss_fc, true_set=gess, false_set=gness)[0],
                    )
                )

arocs = pd.DataFrame(arocs)
arocs.to_excel(f"{rpath}/Libraries_guides_subsample.xlsx", index=False)


# KS optimal library Top 2 and Top 3

lib_metrics = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0)
lib_metrics = lib_metrics[lib_metrics.index.isin(ky_v1_count.index)]

lib_ks = {
    n: lib_metrics.sort_values("ks_control_min", ascending=False)
    .groupby("Gene")
    .head(n=n)
    for n in [2, 3]
}

lib_ks_aucs = {}
for n in [2, 3]:
    lib_ks_aucs[n] = {}

    ks_count = ky_v1_count.loc[lib_ks[n].index]
    ks_fc = ks_count.norm_rpm().foldchange(ky_v1_data.plasmids)
    ks_fc = ks_fc.groupby(lib_ks[n]["Gene"]).mean().mean(1)

    for m in ["aroc", "aupr", "fpr"]:
        if m == "aroc":
            auc = QCplot.pr_curve(ks_fc)

        elif m == "aupr":
            auc = QCplot.recall_curve(ks_fc)[2]

        elif m == "fpr":
            auc = QCplot.aroc_threshold(ks_fc)[0]

        lib_ks_aucs[n][m] = auc


# Plot subsample of replicates and sgRNAs on A375 cell line across GeCKOv2, Brunello and KosukeYusa V1.0

f, axs = plt.subplots(
    3,
    len(datasets),
    sharex="col",
    sharey="all",
    figsize=(len(datasets) * 2.0, 4.5),
    dpi=600,
)

for j, metric in enumerate(["aroc", "aupr", "fpr"]):
    for i, dset in enumerate(datasets):
        ax = axs[j, i]

        df = arocs.query(
            f"(library == '{dset['name']}') & (description == '{dset['description']}')"
        )

        pal = sns.light_palette(
            clib_palette[dset["name"]], n_colors=df["n_samples"].max()
        ).as_hex()

        sns.boxplot(
            "nguides",
            metric,
            "n_samples",
            df,
            ax=ax,
            palette=pal,
            saturation=1,
            showcaps=False,
            boxprops=dict(linewidth=0.3),
            whiskerprops=dict(linewidth=0.3),
            flierprops=CrispyPlot.FLIERPROPS,
            notch=True,
        )

        if dset["name"] == "Yusa_v1":
            lib_ks_pal = {2: "#e6550d", 3: "#fdae6b"}

            for n in lib_ks:
                ax.axhline(
                    lib_ks_aucs[n][metric],
                    ls="-",
                    lw=0.5,
                    c=lib_ks_pal[n],
                    zorder=0,
                    label=f"Top {n}",
                )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

        ax.set_title(f"{dset['name']} {dset['description']}" if j == 0 else None)
        ax.set_xlabel("Number of guides" if j == 1 else None)
        ax.set_ylabel(metric if i == 0 else None)

        ax.legend(
            loc=4, frameon=False, prop={"size": 4}, title="Replicates"
        ).get_title().set_fontsize("4")

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(f"{rpath}/guides_libraries_benchmark.pdf", bbox_inches="tight")
plt.close("all")
