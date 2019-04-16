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
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from CRISPRData import ReadCounts
from crispy.Utils import Utils
from pyensembl import EnsemblRelease
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from crispy.GuideDesign import GuideDesign
from crispy.QCPlot import QCplot
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, mannwhitneyu, ks_2samp, spearmanr

rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

# -
gene = "PPARGC1B"

# -
ky_counts = CRISPRDataSet("Yusa_v1.1")

essential_sgrnas = Utils.get_sanger_essential()
essential_sgrnas = set(essential_sgrnas[essential_sgrnas["ADM_PanCancer_CF"]]["Gene"])
essential_sgrnas = {
    i for i in ky_counts.counts.index if i.split("_")[0] in essential_sgrnas
}

non_essential_sgrnas = Utils.get_non_essential_genes(return_series=False)
non_essential_sgrnas = {
    i for i in ky_counts.counts.index if i.split("_")[0] in non_essential_sgrnas
}

crtl_sgrnas = {i for i in ky_counts.counts.index if i.startswith("CTRL0")}


ky_fc = (
    ky_counts.counts.remove_low_counts(ky_counts.plasmids)
    .norm_rpm()
    .foldchange(ky_counts.plasmids)
    .scale(essential=essential_sgrnas, non_essential=crtl_sgrnas)
)

# -
avana_guides = [i for i in ky_fc.index if i.startswith("sg")]
ky_fc = ky_fc.drop(avana_guides)


# -
crtl_sgrnas_fc = ky_fc.reindex(crtl_sgrnas).median(1).dropna()
ness_sgrnas_fc = ky_fc.reindex(non_essential_sgrnas).median(1).dropna()
ess_sgrnas_fc = ky_fc.reindex(essential_sgrnas).median(1).dropna()


# -
plt.figure(figsize=(2.5, 2), dpi=600)

sns.distplot(
    crtl_sgrnas_fc, hist=False, label="Control", kde_kws={"cut": 0}, color="#3182bd"
)
sns.distplot(
    ness_sgrnas_fc,
    hist=False,
    label="Non-essential",
    kde_kws={"cut": 0},
    color="#31a354",
)
sns.distplot(
    ess_sgrnas_fc, hist=False, label="Essential", kde_kws={"cut": 0}, color="#f03b20"
)

plt.xlabel("Fold-change")
plt.legend(frameon=False)

plt.savefig(f"{rpath}/fc_sgrnas_distplot.png", bbox_inches="tight", dpi=600)
plt.close("all")


# -
ks_sgrna = []
# sgrna = "TP53_CCDS11118.1_ex6_17:7578517-7578540:-_5-3"
for sgrna in ky_fc.index:
    d_ctrl, p_ctrl = ks_2samp(ky_fc.loc[sgrna], crtl_sgrnas_fc)
    d_ness, p_ness = ks_2samp(ky_fc.loc[sgrna], ness_sgrnas_fc)

    ks_sgrna.append(
        {
            "sgrna": sgrna,
            "ks_ctrl": d_ctrl,
            "pval_ctrl": p_ctrl,
            "ks_ness": d_ness,
            "pval_ness": p_ness,
        }
    )

ks_sgrna = pd.DataFrame(ks_sgrna).set_index("sgrna")
ks_sgrna["median_fc"] = ky_fc.loc[ks_sgrna.index].median(1).values
ks_sgrna["gene"] = ky_counts.lib.loc[ks_sgrna.index, "GENES"].values

# -
x, y = ks_sgrna["ks_ctrl"], ks_sgrna["median_fc"]

data, x_e, y_e = np.histogram2d(x, y, bins=20)
z = interpn(
    (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
    data,
    np.vstack([x, y]).T,
    method="splinef2d",
    bounds_error=False,
)

plt.figure(figsize=(2.5, 2), dpi=600)
g = plt.scatter(
    x,
    y,
    c=z,
    marker="o",
    edgecolor="",
    cmap="Spectral_r",
    s=1,
    alpha=0.85,
)
plt.colorbar(g, spacing="uniform", extend="max")
plt.xlabel("K-S statistic")
plt.ylabel("Median fold-change")
plt.title("Test sgRNAs are drawn from\nnon-targeting guides distribution")
plt.savefig(f"{rpath}/ks_median_jointplot.png", bbox_inches="tight", dpi=600)
plt.close("all")


# -
sgrnas_lists = [
    ("control", "#3182bd", crtl_sgrnas),
    ("essential", "#f03b20", essential_sgrnas),
    ("non-essential", "#31a354", non_essential_sgrnas),
    ("tumour-suppressors", "#756bb1", set(ks_sgrna[ks_sgrna["gene"].isin(["TP53", "PTEN"])].index))
]

for (n, c, sgrnas) in sgrnas_lists:
    plt.figure(figsize=(2, 2), dpi=600)

    plt.scatter(
        ks_sgrna["ks_ctrl"], ks_sgrna["median_fc"],
        marker="o",
        edgecolor="",
        color=CrispyPlot.PAL_DBGD[2],
        s=1,
        alpha=0.5,
        label='all',
    )

    plt.scatter(
        ks_sgrna.loc[sgrnas, "ks_ctrl"], ks_sgrna.loc[sgrnas, "median_fc"],
        marker="o",
        edgecolor="",
        color=c,
        s=1,
        alpha=0.75,
        label=n
    )

    plt.xlabel("K-S statistic")
    plt.ylabel("Median fold-change")
    plt.title("Test sgRNAs are drawn from\nnon-targeting guides distribution")
    plt.legend(frameon=False)

    plt.savefig(f"{rpath}/ks_median_jointplot_{n}.png", bbox_inches="tight", dpi=600)
    plt.close("all")


# -
ky_fc_genes = ky_fc.groupby(ky_counts.lib.loc[ky_fc.index, "GENES"].dropna()).mean()
_, original_stats = QCplot.plot_cumsum_auc(ky_fc_genes, Utils.get_essential_genes())
plt.close("all")


filtered_guides = (
    ks_sgrna.sort_values("ks_ctrl", ascending=False).dropna().groupby("gene").head(2)
)

ky_fc_filtered_genes = ky_fc.loc[filtered_guides.index]
ky_fc_filtered_genes = ky_fc_filtered_genes.groupby(
    ky_counts.lib.loc[ky_fc_filtered_genes.index, "GENES"].dropna()
).mean()

_, filtered_stats = QCplot.plot_cumsum_auc(
    ky_fc_filtered_genes, Utils.get_essential_genes()
)
plt.close("all")


# -
ess_aucs = pd.concat(
    [
        pd.Series(original_stats["auc"]).rename("original"),
        pd.Series(filtered_stats["auc"]).rename("filtered"),
    ],
    axis=1,
    sort=False,
)

plt.figure(figsize=(2.0, 2.0), dpi=600)
QCplot.aucs_scatter(x="original", y="filtered", data=ess_aucs, marker="o")
plt.xlabel("All guides (AURC)")
plt.ylabel("Top 2 guides (AURC)")
plt.title("Recall essential genes (Hart et al.)\nKosuke v1.1")
plt.savefig(f"{rpath}/ess_aurc_k_v1.1.png", bbox_inches="tight", dpi=600)
plt.close("all")


# -
ky_v1_counts = CRISPRDataSet("Yusa_v1")

ky_v1_fc = (
    ky_v1_counts.counts.remove_low_counts(ky_v1_counts.plasmids)
    .norm_rpm()
    .foldchange(ky_v1_counts.plasmids)
)

# All guides
ky_v1_fc_genes = ky_v1_fc.groupby(
    ky_v1_counts.lib.loc[ky_v1_fc.index, "Gene"].dropna()
).mean()
ky_v1_fc_genes = ReadCounts(ky_v1_fc_genes).scale()

_, original_v1_stats = QCplot.plot_cumsum_auc(
    ky_v1_fc_genes, Utils.get_essential_genes()
)
plt.close("all")

# Filtered guides
ky_v1_fc_filtered_genes = ky_v1_fc.loc[filtered_guides.index].dropna()
ky_v1_fc_filtered_genes = ky_v1_fc_filtered_genes.groupby(
    ky_v1_counts.lib.loc[ky_v1_fc_filtered_genes.index, "Gene"].dropna()
).mean()
ky_v1_fc_filtered_genes = ReadCounts(ky_v1_fc_filtered_genes).scale()

_, v1_filtered_stats = QCplot.plot_cumsum_auc(
    ky_v1_fc_filtered_genes, Utils.get_essential_genes()
)
plt.close("all")

# Plot
v1_ess_aucs = pd.concat(
    [
        pd.Series(original_v1_stats["auc"]).rename("original"),
        pd.Series(v1_filtered_stats["auc"]).rename("filtered"),
    ],
    axis=1,
    sort=False,
)

plt.figure(figsize=(2, 2), dpi=600)
QCplot.aucs_scatter(x="original", y="filtered", data=v1_ess_aucs, marker="o")
plt.xlabel("All guides (AURC)")
plt.ylabel("Top 2 guides (AURC)")
plt.title("Recall essential genes (Hart et al.)\nKosuke v1.0")
plt.savefig(f"{rpath}/ess_aurc_k_v1.0.png", bbox_inches="tight", dpi=600)
plt.close("all")

# Random guides
random_guides = ks_sgrna.sample(frac=1).dropna().groupby("gene").head(2)

ky_v1_fc_random_genes = ky_v1_fc.loc[random_guides.index].dropna()
ky_v1_fc_random_genes = ky_v1_fc_random_genes.groupby(
    ky_v1_counts.lib.loc[ky_v1_fc_random_genes.index, "Gene"].dropna()
).mean()
ky_v1_fc_random_genes = ReadCounts(ky_v1_fc_random_genes).scale()

_, v1_random_stats = QCplot.plot_cumsum_auc(
    ky_v1_fc_random_genes, Utils.get_essential_genes()
)
plt.close("all")

# Plot
v1_ess_aucs = pd.concat(
    [
        pd.Series(original_v1_stats["auc"]).rename("original"),
        pd.Series(v1_random_stats["auc"]).rename("random"),
    ],
    axis=1,
    sort=False,
)

plt.figure(figsize=(2, 2), dpi=600)
QCplot.aucs_scatter(x="original", y="random", data=v1_ess_aucs, marker="o")
plt.xlabel("All guides (AURC)")
plt.ylabel("Random 2 guides (AURC)")
plt.title("Recall essential genes (Hart et al.)\nKosuke v1.0")
plt.savefig(f"{rpath}/ess_aurc_k_v1.0_random.png", bbox_inches="tight", dpi=600)
plt.close("all")


# -
genes = set(ky_v1_fc_filtered_genes.index).intersection(ky_v1_fc_genes.index)
samples = set(ky_v1_fc_filtered_genes).intersection(ky_v1_fc_genes)

gene_corr = {}
for g in genes:
    r, p = spearmanr(
        ky_v1_fc_filtered_genes.loc[g, samples], ky_v1_fc_genes.loc[g, samples]
    )
    gene_corr[g] = dict(spearmanr=r, pval=p)
gene_corr = pd.DataFrame(gene_corr).T.sort_values("spearmanr")

drive_genes = pd.read_csv(
    f"{pkg_resources.resource_filename('crispy', 'data/gene_sets/')}/driver_genes_2018-09-13_1322.csv"
)

plt.figure(figsize=(2, 2), dpi=600)
sns.distplot(
    gene_corr["spearmanr"],
    hist=False,
    kde_kws=dict(cut=0, shade=True),
    color=CrispyPlot.PAL_DBGD[2],
    label="all",
)
sns.distplot(
    gene_corr.loc[drive_genes["gene_symbol"]]["spearmanr"].dropna(),
    hist=False,
    kde_kws=dict(cut=0, shade=True),
    color=CrispyPlot.PAL_DBGD[1],
    label="Driver genes",
)
plt.xlabel("FC correlation (spearman)")
plt.title("Compare gene fold-change\nAll vs top 2")
plt.legend(frameon=False)
plt.savefig(f"{rpath}/gene_fc_corr_distplot_k_v1.png", bbox_inches="tight", dpi=600)
plt.close("all")


# -
gd = GuideDesign()
transcripts = gd.get_wes_transcripts(gene)

exons = ["ENSE00001845064"]
sgrnas = gd.get_wes_sgrnas(exons)

ensembl = EnsemblRelease(94)

transcript = ensembl.transcript_by_id("ENST00000309241")
sgrna = sgrnas.iloc[0][5]

gd.reverse_complement(sgrna["seq"]) in transcript.sequence
sgrna["seq"] in transcript.sequence
