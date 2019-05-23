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

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from C50K import clib_palette
from crispy.QCPlot import QCplot
from itertools import combinations
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet, Library


# -
log = logging.getLogger("Crispy")
dpath = pkg_resources.resource_filename("crispy", "data/")
rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")


# CRISPR Project Score - Kosuke Yusa v1.1

ky_v1_data = CRISPRDataSet("Yusa_v1")
ky_v1_fc = (
    ky_v1_data.counts.remove_low_counts(ky_v1_data.plasmids)
    .norm_rpm()
    .foldchange(ky_v1_data.plasmids)
)
ky_v1_A375 = ky_v1_fc[[c for c in ky_v1_fc if c.startswith("A375")]]
ky_v1_library = ky_v1_data.lib.loc[ky_v1_A375.index, "Gene"]


# Brunello

brunello_data = CRISPRDataSet("Brunello_A375")
brunello_A375 = (
    brunello_data.counts.remove_low_counts(brunello_data.plasmids)
    .norm_rpm()
    .foldchange(brunello_data.plasmids)
)
brunello_library = brunello_data.lib.loc[brunello_A375.index, "Gene"]
brunello_library = brunello_library[
    [not v.startswith("NO_CURRENT_") for v in brunello_library]
]


# GeCKO v2

gecko_data = CRISPRDataSet("GeCKOv2")
gecko_A375 = (
    gecko_data.counts.remove_low_counts(gecko_data.plasmids)
    .norm_rpm()
    .foldchange(gecko_data.plasmids)
)
gecko_A375 = gecko_A375[[c for c in gecko_A375 if c.startswith("A375")]]
gecko_library = gecko_data.lib[gecko_data.lib.index.isin(gecko_A375.index)]["Gene"]
gecko_library = gecko_library[[not v.startswith("NonTargeting") for v in gecko_library]]
gecko_library = gecko_library[[not v.startswith("hsa-") for v in gecko_library]]


#

datasets = [
    dict(name="GeCKOv2", description="v2", dataset=gecko_A375, library=gecko_library),
    dict(
        name="Yusa_v1",
        description="Day 14",
        dataset=ky_v1_A375[ky_v1_A375.columns[:3]],
        library=ky_v1_library,
    ),
    dict(
        name="Yusa_v1",
        description="Score",
        dataset=ky_v1_A375[ky_v1_A375.columns[-3:]],
        library=ky_v1_library,
    ),
    dict(
        name="Brunello",
        description="Original tracrRNA",
        dataset=brunello_A375[brunello_A375.columns[:2]],
        library=brunello_library,
    ),
    dict(
        name="Brunello",
        description="Modified tracrRNA",
        dataset=brunello_A375[brunello_A375.columns[-3:]],
        library=brunello_library,
    ),
]


#

ky_v11_ks = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0)
ky_v11_ks_lib = ky_v11_ks.sort_values("ks_control", ascending=False).groupby("Gene").head(n=2)


#

arocs = []
n_iterations = 10

for dset in datasets:
    lib = dset["library"]

    # Library
    gene_nguides = lib.value_counts()
    nguides_thres = int(gene_nguides.median())

    guides = lib[lib.isin(gene_nguides[gene_nguides >= nguides_thres].index)]

    # Iterate number of guides
    for nguides in np.arange(1, nguides_thres + 1):

        # Bootstrap
        for i in range(n_iterations):
            log.info(
                f"{dset['name']} - {dset['description']} - #(guides) = {nguides} : iteration {i}"
            )

            # Samples
            nsamples = dset["dataset"].shape[1]
            samples = [
                s
                for n in np.arange(1, nsamples + 1)
                for s in combinations(dset["dataset"], n)
            ]

            # Guides
            guides_rand = guides.groupby(guides).apply(lambda x: x.sample(n=nguides))
            guides_rand.index = guides_rand.index.droplevel()

            # Essential/Non-essential benchmark
            for ss in samples:
                ss_fc = (
                    dset["dataset"].loc[guides_rand.index].groupby(guides_rand).mean()
                )
                ss_fc = ss_fc[list(ss)].mean(1)

                arocs.append(
                    dict(
                        library=dset["name"],
                        description=dset["description"],
                        samples=";".join(ss),
                        n_samples=len(ss),
                        nguides=nguides,
                        aroc=QCplot.pr_curve(ss_fc),
                        aupr=QCplot.recall_curve(ss_fc)[2],
                    )
                )

arocs = pd.DataFrame(arocs)
arocs.to_excel(f"{rpath}/Libraries_guides_subsample.xlsx", index=False)


#

f, axs = plt.subplots(
    2,
    len(datasets),
    sharex="col",
    sharey="all",
    figsize=(len(datasets) * 2.0, 3.0),
    dpi=600,
)

for j, metric in enumerate(["aroc", "aupr"]):
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
            "aroc" if j == 0 else "aupr",
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
            ks_metric = dset["dataset"].reindex(ky_v11_ks_lib.index)
            ks_metric = ks_metric.groupby(ky_v11_ks_lib["Gene"]).mean().mean(1).dropna()
            ks_metric = QCplot.pr_curve(ks_metric) if j == 0 else QCplot.recall_curve(ks_metric)[2]

            ax.axhline(
                ks_metric,
                ls="-",
                lw=1,
                c=clib_palette["minimal"],
                zorder=0,
            )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        ax.set_title(f"{dset['name']} {dset['description']}" if j == 0 else None)
        ax.set_xlabel("Number of guides" if j == 1 else None)
        ax.set_ylabel(("ROC-AUC" if j == 0 else "AUPR") if i == 0 else None)

        ax.legend(
            loc=4, frameon=False, prop={"size": 4}, title="Replicates"
        ).get_title().set_fontsize("4")

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(f"{rpath}/guides_libraries_benchmark.png", bbox_inches="tight")
plt.close("all")
