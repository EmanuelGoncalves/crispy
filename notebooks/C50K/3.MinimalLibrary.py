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

import pandas as pd
from crispy.QCPlot import QCplot
from C50K import LOG, rpath, dpath
from crispy.CRISPRData import CRISPRDataSet, Library


# KY v1.1 library

ky_v11_lib = Library.load_library("Yusa_v1.1.csv.gz")
sabatini_lib = Library.load_library("Sabatini_Lander_v3.csv.gz")
avana_lib = Library.load_library("Avana_v1.csv.gz")
brunello_lib = Library.load_library("Brunello_v1.csv.gz")


# Brunello RootDoenchScores

brunello_scores = pd.read_csv(f"{dpath}/jacks/Brunello_RuleSet2.csv").set_index("sgRNA Target Sequence")


# Protein coding genes
pgenes = pd.read_csv(f"{dpath}/protein_coding_genes.txt", sep="\t")


# Gene symbol map

hgnc_map = pd.read_csv(f"{dpath}/crispr_libs/Yusa_v1.1_HGNC_ID_update.csv")


# KY v1.0 & v1.1 sgRNA metrics

ky_v11_metrics = pd.read_excel(
    f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0
).sort_values("ks_control_min")


# CRISPR Project Score - Kosuke Yusa v1.1

ky_v11_data = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_data.counts.remove_low_counts(ky_v11_data.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)


# Broad DepMap19Q2 - Avana

avana_data = CRISPRDataSet("Avana")
avana_fc = (
    avana_data.counts.remove_low_counts(avana_data.plasmids)
    .norm_rpm()
    .foldchange(avana_data.plasmids)
)


# Avana sgRNA metrics

avana_metrics = pd.read_excel(
    f"{rpath}/Avana_sgRNA_metrics.xlsx", index_col=0
).sort_values("ks_control_min")


# JACKS v1.1
jacks_v11 = pd.read_csv(f"{dpath}/jacks/Yusa_v1.1.csv", index_col=0)
ky_v11_metrics["jacks_v11"] = jacks_v11.reindex(ky_v11_metrics.index)["X1"].values
ky_v11_metrics["jacks_v11_min"] = abs(ky_v11_metrics["jacks_v11"] - 1)


# Top genes

ks_top2 = ky_v11_metrics.sort_values("ks_control_min").groupby("Gene").head(2)
ks_top3 = ky_v11_metrics.sort_values("ks_control_min").groupby("Gene").head(3)


# Defining a minimal library
# U6 promoter stop codon 5-6 Ts

gene_id_map = dict(
    AGAP7="AGAP7P",
    NAPRT1="NAPRT",
    PNMA6C="PNMA6A",
    PTPN20A="PTPN20",
)

minimal_lib = []
for g, df in ky_v11_metrics.groupby("Gene"):
    g_guides = df[(df[["jacks_min", "jacks_v11_min"]] < 1.5).sum(1) >= 1].head(2)
    g_guides = g_guides.assign(info="KosukeYusa")
    g_map = gene_id_map[g] if g in gene_id_map else g

    if len(g_guides) < 2:
        g_guides_avana = avana_metrics.query(f"Gene == '{g_map}'")
        g_guides_avana = g_guides_avana[g_guides_avana["jacks_min"] < 1.5].head(2)
        g_guides_avana = g_guides_avana.assign(info="Avana")
        g_guides = pd.concat([g_guides, g_guides_avana], sort=False).sort_values("ks_control_min").head(2)

    if len(g_guides) < 2:
        g_guides_brunello = brunello_scores[brunello_scores["Target Gene Symbol"] == g_map]
        g_guides_brunello = g_guides_brunello[g_guides_brunello["Rule Set 2 score"] > 0.4]
        g_guides_brunello = g_guides_brunello.sort_values("Rule Set 2 score", ascending=False)
        g_guides_brunello = g_guides_brunello.assign(info="Brunello")
        g_guides = pd.concat([g_guides, g_guides_brunello.head(2 - g_guides.shape[0])], sort=False)

    if len(g_guides) < 2:
        LOG.warning(f"Failed: {g} ({df.shape[0]})")

    minimal_lib.append(g_guides)

minimal_lib = pd.concat(minimal_lib)
minimal_lib["info"] = ["LanderSabatini" if k[:2] == "sg" else v for k, v in minimal_lib["info"].iteritems()]
minimal_lib["info"].value_counts()


#

top2_fc = ky_v11_fc.loc[ks_top2.index].groupby(ks_top2["Gene"]).mean()
minl_fc = ky_v11_fc.reindex(minimal_lib.index).groupby(ks_top2["Gene"]).mean()

top2_aurc = pd.Series({c: QCplot.pr_curve(top2_fc[c]) for c in top2_fc})
minl_aurc = pd.Series({c: QCplot.pr_curve(minl_fc[c]) for c in minl_fc})

aurc = pd.concat([top2_aurc.rename("top2"), minl_aurc.rename("minl")], axis=1)
