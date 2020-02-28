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
import matplotlib.patches as mpatches
from natsort import natsorted
from crispy.GIPlot import GIPlot
from scipy.stats import mannwhitneyu
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from crispy.Utils import Utils
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    CopyNumber,
    Methylation,
    Sample,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH proteomics
#

prot = Proteomics().filter(perc_measures=None)
LOG.info(f"Proteomics: {prot.shape}")


# Gene expression
#

gexp = GeneExpression().filter(subset=list(prot))
LOG.info(f"Transcriptomics: {gexp.shape}")


# Copy number
#

cnv = CopyNumber().filter(subset=list(prot))
LOG.info(f"Copy number: {cnv.shape}")


# Methylation
#

methy = Methylation().filter(subset=list(prot))
LOG.info(f"Methylation: {methy.shape}")


# Overlaps
#

ss = Sample().samplesheet

samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
genes = list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Copy-number amplifications
#

cnv_amp = pd.DataFrame(
    {
        s: {
            g: CopyNumber.is_amplified(cnv.loc[g, s], ss.loc[s, "ploidy"])
            for g in genes
        }
        for s in samples
    }
)

cnv_amp_ctype = cnv_amp.sum().groupby(ss.loc[samples, "cancer_type"]).median()
cnv_amp_gene = cnv_amp.groupby(ss.loc[samples, "cancer_type"], axis=1).sum()


# Correlations with copy number
#


def corr_genes(g, context):
    LOG.info(f"Gene={g}; Context={context}")
    m = (
        pd.concat(
            [
                prot.loc[g].rename("proteomics"),
                gexp.loc[g].rename("transcriptomics"),
                cnv.loc[g].rename("copynumber"),
            ],
            axis=1,
            sort=False,
        )
        .loc[samples]
        .dropna()
    )

    prot_r_pancancer, _ = spearmanr(m["copynumber"], m["proteomics"])
    gexp_r_pancancer, _ = spearmanr(m["copynumber"], m["transcriptomics"])

    m = m.reindex(ss_ct[context]).dropna()
    prot_r_context, _ = spearmanr(m["copynumber"], m["proteomics"])
    gexp_r_context, _ = spearmanr(m["copynumber"], m["transcriptomics"])

    res = dict(
        gene=g,
        context=context,
        len=m.shape[0],
        amp=cnv_amp_gene.loc[g, context],

        prot_r_context=prot_r_context,
        gexp_r_context=gexp_r_context,
        att_context=gexp_r_context - prot_r_context,

        prot_r_pancancer=prot_r_pancancer,
        gexp_r_pancancer=gexp_r_pancancer,
        att_pancancer=gexp_r_pancancer - prot_r_pancancer,
    )

    return res


ss_ct = ss.loc[samples].reset_index().groupby("cancer_type")["model_id"].agg(set)
ss_ct = pd.Series({t: s for t, s in ss_ct.iteritems() if len(s) >= 15})

patt = pd.DataFrame([corr_genes(g, c) for g in genes for c in ss_ct.index]).dropna()
# patt["amp"] = [cnv_amp_gene.loc[g, c] for g, c in patt[["gene", "context"]].values]


#
#

gene = "ERBB2"

plot_df = (
    pd.concat(
        [
            cnv.loc[gene].rename("cn"),
            gexp.loc[gene].rename("gexp"),
            prot.loc[gene].rename("prot"),
            ss[["ploidy", "tissue", "cancer_type"]],
        ],
        axis=1,
        sort=False,
    )
    .dropna()
    .sort_values("cn", ascending=False)
)
plot_df["amp"] = [
    CopyNumber.is_amplified(c, p) for c, p in plot_df[["cn", "ploidy"]].values
]

for dtype in ["gexp", "prot"]:
    ax = GIPlot.gi_classification(
        "amp", dtype, plot_df, orient="v", palette=GIPlot.PAL_DTRACE
    )

    t, p = mannwhitneyu(
        plot_df.query("amp == 1")[dtype], plot_df.query("amp == 0")[dtype]
    )
    ax.text(
        0.95,
        0.05,
        f"Mann-Whitney U\np-value = {p:.2g}",
        fontsize=4,
        transform=ax.transAxes,
        ha="right",
    )

    ax.set_ylabel(f"{dtype}")
    ax.set_xlabel(f"Amplified")
    ax.set_title(gene)

    plt.gcf().set_size_inches(0.7, 1.5)
    plt.savefig(
        f"{RPATH}/2.patt_amp_boxplot_{gene}_{dtype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


#
#

plot_df = patt.query("len > 20")

order = list(
    plot_df.groupby("context")["att_context"].median().sort_values(ascending=False).index
)

fig, ax = plt.subplots(1, 1, figsize=(2, 0.15 * len(order)), dpi=600)

sns.boxplot(
    x="att_context",
    y="context",
    data=plot_df,
    order=order,
    orient="h",
    notch=True,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=CrispyPlot.PAL_CANCER_TYPE,
    showcaps=False,
    ax=ax,
    saturation=1.0,
)

ax.grid(True, axis="x", ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{RPATH}/2.patt_cancertypes_boxplot.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


#
#

plot_df = patt.query("len > 15")
plot_df["amp"] = [Utils.bin_cnv(v, 5) for v in plot_df["amp"]]

order = natsorted(set(plot_df["amp"]))

_, ax = plt.subplots(1, 1, figsize=(4, 0.05 * len(set(plot_df["context"])) * len(order)), dpi=600)

sns.boxplot(
    x="att_context",
    y="amp",
    hue="context",
    order=order,
    data=plot_df,
    orient="h",
    notch=True,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=CrispyPlot.PAL_CANCER_TYPE,
    showcaps=False,
    ax=ax,
    saturation=1.0,
)

ax.grid(True, axis="x", ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.savefig(
    f"{RPATH}/2.patt_gene_context_att.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


#
#

plot_df = patt.query("len > 15").groupby("context")[["att_context", "amp", "len"]].agg({"att_context": np.median, "amp": np.sum, "len": np.median})

for dtype in ["amp", "len"]:
    GIPlot.gi_regression(dtype, "att_context", plot_df)
    plt.savefig(
        f"{RPATH}/2.patt_context_att_{dtype}.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")
