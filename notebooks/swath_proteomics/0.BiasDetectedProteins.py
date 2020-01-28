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
from crispy.QCPlot import QCplot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
from sklearn.metrics import roc_auc_score, roc_curve
from crispy.DataImporter import Proteomics, CORUM, GeneExpression


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH-Proteomics
#

prot = Proteomics().filter(perc_measures=None, replicate_thres=None)
LOG.info(f"Proteomics: {prot.shape}")


# Transcriptomics
#

gexp = GeneExpression().filter(subset=prot.columns)
LOG.info(f"Transcriptomics: {gexp.shape}")


# Gene-sets
#

pathway = "c5.cc.v7.0.symbols.gmt"
gset = Enrichment(gmts=[pathway], sig_min_len=25, padj_method="fdr_bh", verbose=1)


# Hypergeometric test
#

background = {g for gs, s in gset.gmts[pathway].items() for g in s}
psets = dict(
    detected={p for p in background if p in prot.index},
    missed={p for p in background if p not in prot.index},
)
LOG.info(
    f"Background={len(background)}; Detected={len(psets['detected'])}; Missed={len(psets['missed'])}"
)

penr = pd.concat(
    [
        gset.hypergeom_enrichments(psets[n], background, pathway)
        .assign(dtype=n)
        .reset_index()
        for n in psets
    ],
    ignore_index=True,
)

penr.to_csv(
    f"{RPATH}/0.Protein_detection_bias_enriched.csv.gz", index=False, compression="gzip"
)
LOG.info(penr[penr["adj.p_value"] < 0.01].query("dtype == 'missed'"))


# Plot missed enrichments
#

for n in psets:
    plot_df = penr[penr["adj.p_value"] < 0.01].query(f"dtype == '{n}'").head(30)
    plot_df["gset"] = [i[3:].lower().replace("_", " ") for i in plot_df["gset"]]

    _, ax = plt.subplots(1, 1, figsize=(2.0, 5.0), dpi=600)

    sns.barplot(
        -np.log10(plot_df["adj.p_value"]),
        plot_df["gset"],
        orient="h",
        color=CrispyPlot.PAL_DTRACE[2],
        ax=ax,
    )

    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(
            -np.log10(row["adj.p_value"]),
            i,
            f"{row['len_intersection']}/{row['len_sig']}",
            va="center",
            ha="left",
            fontsize=5,
            zorder=10,
            color=CrispyPlot.PAL_DTRACE[2],
        )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
    ax.set_xlabel("Hypergeometric test (-log10 FDR)")
    ax.set_ylabel("")
    ax.set_title(f"GO celluar component enrichment - {n.capitalize()} proteins")

    plt.savefig(
        f"{RPATH}/0.Detection_bias_{n}_barplot.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


# Protein length and mass
#

pmass = pd.read_csv(f"{DPATH}/protein_mass.tab.gz", sep="\t")
pmass = pmass.assign(Measured=pmass["Gene names  (primary )"].isin(prot.index).astype(int))
pmass["Measured"] = pmass["Measured"].apply(lambda v: "Yes" if v else "No")
pmass["Mass"] = pmass["Mass"].apply(lambda v: int(v.replace(",", "")))

for m in ["Mass", "Length"]:
    fig, ax = plt.subplots(1, 1, figsize=(1, 2), dpi=600)
    sns.boxplot(
        pmass["Measured"],
        np.log(pmass[m]),
        ax=ax,
        order=["No", "Yes"],
        notch=True,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        medianprops=CrispyPlot.MEDIANPROPS,
        flierprops=CrispyPlot.FLIERPROPS,
        palette=CrispyPlot.PAL_YES_NO,
        showcaps=False,
        saturation=1.0,
    )
    ax.set_xlabel("Measured")
    ax.set_ylabel(f"Mass (log Da)" if m == "Mass" else "Length (log AA)")
    ax.grid(True, axis="y", ls="-", lw=0.1, alpha=1.0, zorder=0)
    plt.savefig(
        f"{RPATH}/0.Detection_bias_protein_{m}.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")


# Protein detection bias by transcript abundance
#

plot_df = gexp.mean(1).rename("mrna").reset_index()
plot_df = plot_df.assign(Measured=plot_df["index"].isin(prot.index).astype(int))
plot_df["Measured"] = plot_df["Measured"].apply(lambda v: "Yes" if v else "No")

fig, ax = plt.subplots(1, 1, figsize=(1, 2), dpi=600)
sns.boxplot(
    plot_df["Measured"],
    plot_df["mrna"],
    ax=ax,
    notch=True,
    order=["No", "Yes"],
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=CrispyPlot.PAL_YES_NO,
    showcaps=False,
    saturation=1.0,
)
ax.set_xlabel("Measured")
ax.set_ylabel(f"mRNA expression (mean voom)")
ax.grid(True, axis="y", ls="-", lw=0.1, alpha=1.0, zorder=0)
plt.savefig(
    f"{RPATH}/0.Detection_bias_mrna.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")
