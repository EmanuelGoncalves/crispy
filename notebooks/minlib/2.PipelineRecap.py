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


# Project Score

ky_v11_top2 = pd.read_csv(
    f"{rpath}/KosukeYusa_v1.1_sgrna_counts_ks_control_min_top2.csv.gz", index_col=0
)

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_count.counts.remove_low_counts(ky_v11_count.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_count.plasmids)
)

ky_v11_metrics = pd.read_excel(
    f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0
)
ky_v11_metrics["top2"] = ky_v11_metrics.index.isin(ky_v11_top2.index).astype(int)


# Strongly Selective Dependencies (SSDs)

ssd_genes = list(pd.read_excel(f"{dpath}/ssd_genes.xls")["CancerGenes"])


# CRISPR pipeline results

cres_min2 = PipelineResults(
    # f"{dpath}/minimal_lib/Top3_KYV11/",
    f"{dpath}/minimal_lib/minlib_top2_dep_v11/",
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


#

sample_thres = []

for s in samples:
    for d, df in [("Top2", cres_min2.fc), ("All", cres_all.fc)]:
        fpr_auc, fpr_thres = QCplot.aroc_threshold(df.loc[genes, s])

        sample_thres.append(dict(
            sample=s, dtype=d, auc=fpr_auc, fpr_thres=fpr_thres
        ))

sample_thres = pd.DataFrame(sample_thres)


#

for value in ["auc", "fpr_thres"]:
    plot_df = pd.pivot_table(sample_thres, index="sample", columns="dtype", values=value)

    plt.figure(figsize=(2.0, 1.5), dpi=600)
    ax = plt.gca()
    sgrnas_scores_scatter(
        df_ks=plot_df,
        x="All",
        y="Top2",
        annot=True,
        diag_line=True,
        reset_lim=False,
        z_method="gaussian_kde",
        ax=ax,
    )
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
    ax.set_xlabel("All sgRNAs")
    ax.set_ylabel("Top 2 sgRNAs")
    ax.set_title(f"{value} 5% FPR")
    plt.savefig(f"{rpath}/recapdep_fpr_{value}_scatter.png", bbox_inches="tight")
    plt.close("all")


#

fc_thres = pd.pivot_table(sample_thres, index="sample", columns="dtype", values="fpr_thres")

fc_bin = {}
for d, df in [("Top2", cres_min2.fc), ("All", cres_all.fc)]:
    fc_bin[d] = {}
    for s in samples:
        fc_bin[d][s] = (df.loc[genes, s] < fc_thres.loc[s, d]).astype(int)
    fc_bin[d] = pd.DataFrame(fc_bin[d])



plot_df = pd.concat([fc_bin[c].sum(1).rename(c) for c in fc_bin], axis=1, sort=False)
plot_df["n_sgrnas"] = ky_v11_count.lib.groupby("Gene")["sgRNA"].count().loc[plot_df.index].values
plot_df["diff"] = plot_df.eval("All - Top2")

plot_df.loc[plot_df.eval("All - Top2").sort_values().index]

plt.figure(figsize=(2.0, 1.5), dpi=600)
ax = plt.gca()
sgrnas_scores_scatter(
    df_ks=plot_df,
    x="All",
    y="Top2",
    annot=True,
    diag_line=True,
    reset_lim=True,
    z_method="gaussian_kde",
    ax=ax,
)
ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
ax.set_xlabel("All sgRNAs")
ax.set_ylabel("Top 2 sgRNAs")
ax.set_title(f"{value} 5% FPR")
plt.savefig(f"{rpath}/recapdep_fpr_fc_bin_scatter.png", bbox_inches="tight")
plt.close("all")


#

g = "RPS9"

g_metrics = ky_v11_metrics.query(f"Gene == '{g}'").sort_values("ks_control_min")

plot_df = ky_v11_fc.loc[g_metrics.index]

row_colors = pd.Series(CrispyPlot.PAL_DBGD)[g_metrics["top2"]]
row_colors.index = g_metrics.index

g = sns.clustermap(
    plot_df,
    cmap="Spectral",
    xticklabels=False,
    figsize=(2, 2),
    row_colors=row_colors,
    center=0,
)
plt.savefig(f"{rpath}/recapdep_sgrna_fc_clustermap.png", bbox_inches="tight", dpi=600)
plt.close("all")


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
                cres_all.mdep_bin.loc[g1, samples], cres_min2.mdep_bin.loc[g2, samples]
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


# Fold-changes benchmark
#
# Spearman orrelation analysis

genes_corr = []
for g in genes:
    c, p = spearmanr(cres_all.fc.loc[g, samples], cres_min2.fc.loc[g, samples])
    genes_corr.append(dict(gene=g, corr=c, pval=p))
genes_corr = pd.DataFrame(genes_corr).sort_values("corr")
genes_corr["ssd_gene"] = genes_corr["gene"].isin(ssd_genes).astype(int)


g = "JAK1"

plot_df = pd.concat(
    [
        cres_all.fc.loc[g].rename("all"),
        cres_min2.fc.loc[g].rename("top2"),
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
ax.set_title(f"{g} fold-change")
plt.savefig(f"{rpath}/recapdep_fc_scatter_{g}.png", bbox_inches="tight")
plt.close("all")


plt.figure(figsize=(2.0, 1.5), dpi=600)
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


# Scatter comparing the total number of gene dependencies

plot_df = pd.concat(
    [
        cres_all.fc_b.loc[genes].sum(1).rename("all"),
        cres_min2.fc_b.loc[genes].sum(1).rename("top2"),
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
ax.set_title("Fold-change - total signif. dependencies\nProject Score - KY v1.1")
plt.savefig(f"{rpath}/recapdep_fc_total_scatter.png", bbox_inches="tight")
plt.close("all")


# MCC of Strongly Selective Dependencies (SSDs) binary dependencies

ssd_mcc = []
for g1 in ssd_genes:
    for g2 in ssd_genes:
        if g1 in genes and g2 in genes:
            mcc = matthews_corrcoef(
                cres_all.fc_b.loc[g1, samples], cres_min2.fc_b.loc[g2, samples]
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
plt.title("Fold-change dependency aggreement\nAll vs top2")
plt.savefig(f"{rpath}/recapdep_fc_mcc_boxplot.png", bbox_inches="tight")
plt.close("all")


# Agreement difference between high and low quality samples
