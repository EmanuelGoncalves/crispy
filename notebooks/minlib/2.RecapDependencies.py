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
from math import sqrt
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from crispy.DataImporter import PipelineResults
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter
from sklearn.metrics import mean_squared_error, matthews_corrcoef


# Manifest

manifest = pd.read_csv(f"{dpath}/minimal_lib/manifest.txt", index_col=0, sep="\t")
manifest["lib"] = [";".join(set(i.split(", "))) for i in manifest["sgRNAlibrary"]]
manifest = manifest.query("lib == 'V1.1'")


# Strongly Selective Dependencies (SSDs)

ssd_genes = list(pd.read_excel(f"{dpath}/ssd_genes.xls")["CancerGenes"])


# CRISPR pipeline results

cres_min2 = PipelineResults(
    f"{dpath}/minimal_lib/minlib_top2_dep/",
    import_fc=True,
    import_bagel=True,
    import_mageck=True,
)

cres_all = PipelineResults(
    f"{dpath}/minimal_lib/KY_dep/",
    import_fc=True,
    import_bagel=True,
    import_mageck=True,
)


# Overlap

genes = list(set(cres_min2.bf_b.index).intersection(cres_all.bf_b.index))
samples = list(
    set(cres_min2.bf_b).intersection(cres_all.bf_b).intersection(manifest.index)
)
LOG.info(f"Genes={len(genes)}; Samples={len(samples)}")


# BAGELr benchmark
#
# Scatter comparing the total number of gene dependencies

plot_df = pd.concat(
    [
        cres_all.bf_b.loc[genes].sum(1).rename("all"),
        cres_min2.bf_b.loc[genes].sum(1).rename("top2"),
    ],
    axis=1,
    sort=False,
).dropna()

plt.figure(figsize=(2.0, 1.5), dpi=600)
ax = plt.gca()
sgrnas_scores_scatter(
    df_ks=plot_df,
    x="all",
    y="top2",
    annot=True,
    diag_line=True,
    reset_lim=True,
    z_method="gaussian_kde",
    ax=ax,
)
ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel("All sgRNAs")
ax.set_ylabel("KS top 2 sgRNAs")
ax.set_title("BAGELr - total signif. dependencies\nProject Score - KY v1.1")
plt.savefig(f"{rpath}/recapdep_bagelr_total_scatter.png", bbox_inches="tight")
plt.close("all")


# MCC of Strongly Selective Dependencies (SSDs) binary dependencies

ssd_mcc = []
for g1 in ssd_genes:
    for g2 in ssd_genes:
        if g1 in genes and g2 in genes:
            mcc = matthews_corrcoef(
                cres_all.bf_b.loc[g1, samples], cres_min2.bf_b.loc[g2, samples]
            )

            ssd_mcc.append(dict(gene1=g1, gene2=g2, mcc=mcc, same=int(g1 == g2)))

ssd_mcc = pd.DataFrame(ssd_mcc)

plt.figure(figsize=(1.0, 1.5), dpi=600)
sns.boxplot(
    "same",
    "mcc",
    color=CrispyPlot.PAL_DBGD[0],
    data=ssd_mcc,
    saturation=1,
    showcaps=False,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    flierprops=CrispyPlot.FLIERPROPS,
    notch=True,
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")
plt.xlabel("Same SSD gene")
plt.ylabel("Matthew's correlation\ncoefficient")
plt.title("BAGELr dependency aggreement\nAll vs top2")
plt.savefig(f"{rpath}/recapdep_bagelr_mcc_boxplot.png", bbox_inches="tight")
plt.close("all")


# Fold-changes benchmark
#
# Spearman orrelation analysis

genes_corr = []
for g in genes:
    c, p = spearmanr(cres_all.fc.loc[g, samples], cres_min2.fc.loc[g, samples])
    genes_corr.append(dict(gene=g, corr=c, pval=p))
genes_corr = pd.DataFrame(genes_corr).sort_values("corr")
genes_corr["ssd_gene"] = genes_corr["gene"].isin(ssd_genes).astype(int)

plt.figure(figsize=(2., 1.5), dpi=600)
sns.distplot(
    genes_corr["corr"],
    hist=False,
    label="All",
    kde_kws={"cut": 0, "shade": True},
    color=CrispyPlot.PAL_DBGD[0],
)
sns.distplot(
    genes_corr.query("ssd_gene == 1")["corr"],
    hist=False,
    label="SSD genes",
    kde_kws={"cut": 0, "shade": True},
    color=CrispyPlot.PAL_DBGD[1],
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.legend(frameon=False)
plt.xlabel("Gene spearman correlation")
plt.title("Fold-change aggreement\nAll vs top2")
plt.savefig(f"{rpath}/recapdep_fc_spearman_displot.png", bbox_inches="tight")
plt.close("all")


# MAGeCK benchmark
#
# Scatter comparing the total number of gene dependencies

plot_df = pd.concat(
    [
        cres_all.mdep_bin.loc[genes].sum(1).rename("all"),
        cres_min2.mdep_bin.loc[genes].sum(1).rename("top2"),
    ],
    axis=1,
    sort=False,
).dropna()

plt.figure(figsize=(2.0, 1.5), dpi=600)
ax = plt.gca()
sgrnas_scores_scatter(
    df_ks=plot_df,
    x="all",
    y="top2",
    annot=True,
    diag_line=True,
    reset_lim=True,
    z_method="gaussian_kde",
    ax=ax,
)
ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel("All sgRNAs")
ax.set_ylabel("KS top 2 sgRNAs")
ax.set_title("MAGeCK - total signif. dependencies\nProject Score - KY v1.1")
plt.savefig(f"{rpath}/recapdep_mageck_total_scatter.png", bbox_inches="tight")
plt.close("all")


# MCC of Strongly Selective Dependencies (SSDs) binary dependencies

ssd_mcc = []
for g1 in ssd_genes:
    for g2 in ssd_genes:
        if g1 in genes and g2 in genes:

            mcc = matthews_corrcoef(
                cres_all.bf_b.loc[g1, samples], cres_min2.bf_b.loc[g2, samples]
            )

            ssd_mcc.append(dict(gene1=g1, gene2=g2, mcc=mcc, same=int(g1 == g2)))

ssd_mcc = pd.DataFrame(ssd_mcc)

plt.figure(figsize=(1.0, 1.5), dpi=600)
sns.boxplot(
    "same",
    "mcc",
    color=CrispyPlot.PAL_DBGD[0],
    data=ssd_mcc,
    saturation=1,
    showcaps=False,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    flierprops=CrispyPlot.FLIERPROPS,
    notch=True,
)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")
plt.xlabel("Same SSD gene")
plt.ylabel("Matthew's correlation\ncoefficient")
plt.title("MAGeCK dependency aggreement\nAll vs top2")
plt.savefig(f"{rpath}/recapdep_mageck_mcc_boxplot.png", bbox_inches="tight")
plt.close("all")


# Agreement difference between high and low quality samples
