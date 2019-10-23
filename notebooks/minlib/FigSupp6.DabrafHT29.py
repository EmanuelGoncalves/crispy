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
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib.Utils import density_interpolate


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Libraries
#

NGUIDES, REMOVE_DISCORDANT = 2, True
ml_lib_name = (
    f"MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz"
)
ml_lib = Library.load_library(ml_lib_name).query("Library == 'KosukeYusa'")
ml_lib = ml_lib.loc[[i for i in ml_lib.index if not i.startswith("CTRL0")]]

libraries = dict(
    All=dict(
        name="All",
        lib=Library.load_library("MasterLib_v1.csv.gz").query(
            "Library == 'KosukeYusa'"
        ),
    ),
    Minimal=dict(name="Minimal", lib=ml_lib),
)


# HT-29 CRISPR-Cas9 + Dabrafenib timecourse (Day 8, 10, 14, 18 and 21)
#

dabraf_data = CRISPRDataSet("HT29_Dabraf")
dabraf_count = dabraf_data.counts.remove_low_counts(dabraf_data.plasmids)
dabraf_ss = pd.read_csv(
    f"{DPATH}/crispr_manifests/HT29_Dabraf_samplesheet.csv.gz", index_col="sample"
)


# Calculate and export gene fold-changes
#

for ltype in libraries:
    LOG.info(f"Exporting gene fold-changes: {ltype}")

    lib = libraries[ltype]["lib"]

    l_counts = dabraf_count[dabraf_count.index.isin(lib.index)]

    l_fc = l_counts.norm_rpm().foldchange(dabraf_data.plasmids)
    l_fc = l_fc.groupby(lib["Approved_Symbol"]).mean()

    libraries[ltype]["fc"] = l_fc
    l_fc.to_csv(f"{RPATH}/HT29_Dabraf_gene_fc_{ltype}.csv.gz", compression="gzip")


# Limma differential essential gene fold-changes
#

dabraf_limma = pd.concat(
    [
        pd.read_csv(f"{RPATH}/HT29_Dabraf_limma_{ltype}.csv")
        .assign(library=ltype)
        .rename(columns={"Unnamed: 0": "Approved_Symbol"})
        for ltype in libraries
    ]
)


# Genes overlap
#

genes = set(libraries["All"]["fc"].index).intersection(libraries["Minimal"]["fc"].index)
LOG.info(f"Genes={len(genes)}")


# Limma scatter plots
#

dif_cols = ["Dif10d", "Dif14d", "Dif18d", "Dif21d"]

x_min = dabraf_limma[dif_cols].mean(1).min() * 1.05
x_max = dabraf_limma[dif_cols].mean(1).max() * 1.05

plot_df = pd.DataFrame(
    dict(
        x=dabraf_limma.query(f"library == 'All'")
        .set_index(["Approved_Symbol"])
        .loc[genes, dif_cols]
        .mean(1),
        y=dabraf_limma.query(f"library == 'Minimal'")
        .set_index(["Approved_Symbol"])
        .loc[genes, dif_cols]
        .mean(1),
    )
).dropna()

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=600)

ax.hexbin(
    plot_df["x"],
    plot_df["y"],
    cmap="Spectral_r",
    gridsize=100,
    mincnt=1,
    bins="log",
    lw=0,
)

rmse = sqrt(mean_squared_error(plot_df["x"], plot_df["y"]))
cor, _ = spearmanr(plot_df["x"], plot_df["y"])
annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"
ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

lims = [x_max, x_min]
ax.plot(lims, lims, "k-", lw=0.3, zorder=0)
ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("Kosuke Yusa V1.1 (5 sgRNAs/Gene)")
ax.set_ylabel("MinLibCas9 (2 sgRNAs/Gene)")
ax.set_title("CRISPR + Dabrafinib (fold-change)")

plt.savefig(
    f"{RPATH}/HT29_Dabraf_fc_scatter.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Fold-change examples

highlights = ["GRB2", "CSK", "EGFR", "ERBB2", "STK11", "SHC1", "NF1", "PTEN"]
conditions = [("DMSO", "#E1E1E1"), ("Dabraf", "#d62728")]

f, axs = plt.subplots(
    2, len(highlights), sharex="all", sharey="all", figsize=(len(highlights) * 2, 4)
)

for i, ltype in enumerate(libraries):
    for j, gene in enumerate(highlights):
        ax = axs[i, j]

        plot_df = pd.concat(
            [libraries[ltype]["fc"].loc[gene], dabraf_ss], axis=1, sort=False
        )
        plot_df = pd.concat(
            [
                plot_df.query("medium != 'Initial'"),
                plot_df.query("medium == 'Initial'").replace(
                    {"medium": {"Initial": "Dabraf"}}
                ),
                plot_df.query("medium == 'Initial'").replace(
                    {"medium": {"Initial": "DMSO"}}
                ),
            ]
        )

        for mdm, clr in conditions:
            plot_df_mrk = plot_df.query(f"medium == '{mdm}'")
            plot_df_mrk = plot_df_mrk.assign(
                time=plot_df_mrk["time"].apply(lambda x: int(x[1:]))
            )
            plot_df_mrk = plot_df_mrk.groupby("time")[gene].agg([min, max, np.mean])
            plot_df_mrk = plot_df_mrk.assign(time=[f"D{i}" for i in plot_df_mrk.index])

            ax.errorbar(
                plot_df_mrk.index,
                plot_df_mrk["mean"],
                c=clr,
                fmt="--o",
                label=mdm,
                capthick=0.5,
                capsize=2,
                elinewidth=1,
                yerr=[
                    plot_df_mrk["mean"] - plot_df_mrk["min"],
                    plot_df_mrk["max"] - plot_df_mrk["mean"],
                ],
            )

            ax.set_xticks(list(plot_df_mrk.index))

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        ax.set_xlabel("Days" if i == 2 else "")
        ax.set_ylabel(f"sgRNA {ltype}\nfold-change" if j == 0 else "")
        ax.set_title(f"{gene}" if i == 0 else "")
        ax.legend(loc=2, frameon=False, prop={"size": 4})

        g_limma = (
            dabraf_limma.query(f"library == '{ltype}'")
            .set_index("Approved_Symbol")
            .loc[gene]
        )
        annot_text = (
            f"Mean FC={g_limma[dif_cols].mean():.1f}; FDR={g_limma['adj.P.Val']:.1e}"
        )
        ax.text(0.05, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="left")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(
    f"{RPATH}/HT29_Dabraf_assoc_examples.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")
