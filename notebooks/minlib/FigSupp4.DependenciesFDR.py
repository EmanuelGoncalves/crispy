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

import ast
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.QCPlot import QCplot
from crispy.DataImporter import Sample
from crispy.CrispyPlot import CrispyPlot
from scipy.stats import spearmanr, pearsonr
from crispy.CRISPRData import CRISPRDataSet, Library, ReadCounts
from minlib.Utils import project_score_sample_map, density_interpolate
from sklearn.metrics import (
    recall_score,
    precision_score,
    average_precision_score,
    precision_recall_curve,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# KY v1.1 library
#

ky_lib = Library.load_library("MasterLib_v1.csv.gz").query("Library == 'KosukeYusa'")
ky_lib = ky_lib[ky_lib.index.isin(ky_counts.index)]


# Minimal library (top 2)
#

NGUIDES, REMOVE_DISCORDANT = 2, True
ml_lib_name = (
    f"MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz"
)
ml_lib = Library.load_library(ml_lib_name).query("Library == 'KosukeYusa'")
ml_lib = ml_lib[ml_lib.index.isin(ky_counts.index)]
ml_lib = ml_lib.loc[[i for i in ml_lib.index if not i.startswith("CTRL0")]]


# Genes overlap
#

genes = set(ky_lib["Approved_Symbol"]).intersection(ml_lib["Approved_Symbol"])
LOG.info(f"Genes={len(genes)}")


# Comparisons
#

stats = []
for fdr_thres in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
    LOG.info(f"FDR thres={fdr_thres}")

    libraries = dict(
        All=dict(name="All", lib=ky_lib), Minimal=dict(name="Minimal", lib=ml_lib)
    )

    # Minimal counts
    #
    for ltype in libraries:
        lib = libraries[ltype]["lib"].copy()

        # sgRNA fold-changes
        fc = ky_counts.copy().loc[lib.index].norm_rpm().foldchange(ky.plasmids)

        # Gene fold-changes
        fc_gene = fc.groupby(lib["Approved_Symbol"]).mean()

        # Gene average fold-changes
        fc_gene_avg = fc_gene.groupby(ky_smap["model_id"], axis=1).mean()

        # Gene average scaled fold-changes
        fc_gene_avg_scl = ReadCounts(fc_gene_avg).scale()

        # FDR threshold
        fpr = pd.DataFrame(
            {
                s: QCplot.aroc_threshold(fc_gene_avg[s], fpr_thres=fdr_thres)
                for s in fc_gene_avg
            },
            index=["auc", "thres"],
        ).T
        fpr["auc"] = [
            QCplot.aroc_threshold(fc_gene_avg[s], fpr_thres=0.2)[0] for s in fpr.index
        ]

        # Gene binarised
        fc_gene_avg_bin = (fc_gene_avg[fpr.index] < fpr["thres"]).astype(int)

        # Store
        libraries[ltype]["fc"] = fc
        libraries[ltype]["fc_gene"] = fc_gene
        libraries[ltype]["fc_gene_avg"] = fc_gene_avg
        libraries[ltype]["fc_gene_avg_scl"] = fc_gene_avg_scl
        libraries[ltype]["fc_gene_avg_bin"] = fc_gene_avg_bin
        libraries[ltype]["fpr"] = fpr

    # Recovered dependencies per cell line
    #
    deprecovered = pd.DataFrame(
        [
            dict(
                samples=s,
                recall=recall_score(
                    libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
                    libraries["Minimal"]["fc_gene_avg_bin"].loc[genes, s],
                ),
                ap=average_precision_score(
                    libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
                    -libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
                ),
                precision=precision_score(
                    libraries["All"]["fc_gene_avg_bin"].loc[genes, s],
                    libraries["Minimal"]["fc_gene_avg_bin"].loc[genes, s],
                ),
                spearman=spearmanr(
                    libraries["All"]["fc_gene_avg"].loc[genes, s],
                    libraries["Minimal"]["fc_gene_avg"].loc[genes, s],
                )[0],
                threshold=libraries["All"]["fpr"].loc[s, "thres"],
            )
            for s in libraries["All"]["fc_gene_avg_bin"]
        ]
    )

    # Stats
    #
    stats.append(deprecovered.assign(fdr=fdr_thres))

stats = pd.concat(stats)

# Plot
#

plot_df = pd.concat(
    [
        stats.groupby("fdr")["ap"].mean().rename("ap_mean"),
        stats.groupby("fdr")["threshold"].mean().rename("threshold_mean"),
        stats.groupby("fdr")["ap"].std().rename("ap_std"),
        stats.groupby("fdr")["threshold"].std().rename("threshold_std"),
    ],
    axis=1,
    sort=False,
)

_, ax = plt.subplots(1, 1, figsize=(2, 2))

ax.errorbar(
    plot_df["ap_mean"],
    plot_df["threshold_mean"],
    xerr=plot_df["ap_std"],
    yerr=plot_df["threshold_std"],
    c=CrispyPlot.PAL_DBGD[0],
    fmt="o",
    lw=0.3,
    mec="white",
    mew=.1,
)

for i, r in plot_df.iterrows():
    ax.text(
        r["ap_mean"] * 1.005,
        r["threshold_mean"],
        f"{(i * 100):.0f}% FDR",
        color="k",
        fontsize=5,
    )

ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

cor, pval = spearmanr(plot_df["ap_mean"], plot_df["threshold_mean"])
annot_text = f"Spearman's R={cor:.2g}, p-val={pval:.1g}"
ax.text(0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right")

ax.set_xlabel("Mean Average Precision (AP) score")
ax.set_ylabel("Mean fold-change FDR threshold")

plt.savefig(
    f"{RPATH}/{ml_lib_name.split('.')[0]}_ap_scatter.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
