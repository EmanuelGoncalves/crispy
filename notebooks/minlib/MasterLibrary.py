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
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib.Utils import define_sgrnas_sets, estimate_ks


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score KY v1.1
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_fc = ky.counts.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)
ky_gsets = define_sgrnas_sets(ky.lib, ky_fc, add_controls=True)
ky_ks = estimate_ks(ky_fc, ky_gsets["nontargeting"]["fc"])


# DepMap 19Q2 Avana
#

avana = CRISPRDataSet("Avana_DepMap19Q2")
avana_fc = (
    avana.counts.remove_low_counts(avana.plasmids).norm_rpm().foldchange(avana.plasmids)
)
avana_gsets = define_sgrnas_sets(
    avana.lib, avana_fc, dataset_name="Avana_DepMap19Q2", add_controls=True
)
avana_ks = estimate_ks(avana_fc, avana_gsets["nontargeting"]["fc"])


# CRISPR-Cas9 libraries
#

brunello = Library.load_library("Brunello_v1.csv.gz")
avana = Library.load_library("Avana_v1.csv.gz")
ky = Library.load_library("Yusa_v1.1.csv.gz")
tkov3 = Library.load_library("TKOv3.csv.gz")


# WGE KY guide sequence match
#

ky_v11_wge = pd.read_csv(f"{DPATH}/update/Yusa_v1.1_WGE_map.txt", sep="\t", index_col=0)
ky_v11_wge = ky_v11_wge[~ky_v11_wge["WGE_sequence"].isna()]
ky_v11_wge["WGE_sequence"] = ky_v11_wge["WGE_sequence"].apply(lambda v: v[:-3]).values


# WGE annotation
#

wge = pd.read_csv(f"{DPATH}/update/MasterList_WGEoutput.txt.gz", sep="\t").drop(
    columns=["Index"]
)
wge = wge.set_index("Master_sgRNA")


# WGE guides genomic coordinates
#

wge_coord = pd.read_csv(f"{DPATH}/update/WGE_coordinates.txt", sep="\t", index_col=0)


# KS scores
#

ks = pd.concat([ky_ks["ks_control"], avana_ks["ks_control"]])


# JACKS scores
#

jacks_v1 = pd.read_csv(f"{DPATH}/jacks/Yusa_v1.0.csv", index_col=0)["X1"]
jacks_v11 = pd.read_csv(f"{DPATH}/jacks/Yusa_v1.1.csv", index_col=0)["X1"]
jacks_ky = pd.concat([jacks_v1, jacks_v11], axis=1, sort=False)
jacks_ky = pd.Series(
    [j1 if abs(j1 - 1) < abs(j2 - 1) else j2 for j1, j2 in jacks_ky.values],
    index=jacks_ky.index,
)

jacks_avana = pd.read_csv(f"{DPATH}/jacks/Avana_v1.csv", index_col=0)["X1"]

jacks = pd.concat([jacks_ky, jacks_avana])


# Rule Set 2
#

ruleset2_ky = pd.read_csv(f"{DPATH}/jacks/Yusa_v1.0_DoenchRoot.csv", index_col=0)[
    "score_with_ppi"
]
ruleset2_brunello = pd.read_csv(
    f"{DPATH}/jacks/Brunello_RuleSet2.csv", index_col="sgRNA Target Sequence"
)["Rule Set 2 score"]

ruleset2 = pd.concat([ruleset2_ky, ruleset2_brunello])


# FORECAST
#

forecast_ky = pd.read_csv(
    f"{DPATH}/jacks/Yusa_v1.0_ForecastFrameshift.csv", index_col=0
)["In Frame Percentage"]


# HGNC gene symbol update
#

hgnc = pd.read_csv(f"{DPATH}/update/hgnc-symbol-check.csv").groupby("Input").first()


# Master library
#

master = (
    pd.read_csv(f"{DPATH}/update/master.csv.gz")
    .drop(columns=["doenchroot", "jacks_min", "ks_control_min"])
    .rename(columns=dict(lib="Library"))
)

master.loc[master["sgRNA_ID"].isin(ky_v11_wge.index), "sgRNA"] = (
    ky_v11_wge.loc[master["sgRNA_ID"], "WGE_sequence"].dropna().values
)

master["Approved_Symbol"] = hgnc.reindex(master["Gene"])["Approved symbol"].values
master["Approved_Name"] = hgnc.reindex(master["Gene"])["Approved name"].values
master["HGNC_ID"] = hgnc.reindex(master["Gene"])["HGNC ID"].values

master["WGE_Sequence"] = (
    wge.reindex(master["sgRNA"])["WGE_Sequence"].replace("not_found", np.nan).values
)
master["WGE_ID"] = (
    wge.reindex(master["sgRNA"])["WGE"].replace("not_found", np.nan).values
)
master["Off_Target"] = (
    wge.reindex(master["sgRNA"])["OT_Summary"].replace("not_found", np.nan).values
)
master["WGE_MatchComment"] = (
    wge.reindex(master["sgRNA"])["Comments"].replace("not_found", np.nan).values
)

master["Chr"] = wge_coord.reindex(master["WGE_ID"].astype(float))["chr_name"].values
master["Start"] = wge_coord.reindex(master["WGE_ID"].astype(float))["chr_start"].values
master["End"] = wge_coord.reindex(master["WGE_ID"].astype(float))["chr_end"].values
master["Strand"] = wge_coord.reindex(master["WGE_ID"].astype(float))["strand"].values
master["Assembly"] = wge_coord.reindex(master["WGE_ID"].astype(float))[
    "assembly"
].values

master["JACKS"] = jacks.reindex(master["sgRNA_ID"]).values
master["RuleSet2"] = ruleset2.reindex(master["sgRNA_ID"]).values
master["FORECAST"] = forecast_ky.reindex(master["sgRNA_ID"]).values
master["KS"] = ks.reindex(master["sgRNA_ID"]).values

master = master.sort_values(
    ["Library", "Approved_Symbol", "KS"], ascending=[True, True, False]
)


# Export Master Library
#

master.drop(columns=["sgRNA", "Gene"]).to_csv(
    f"{DPATH}/crispr_libs/MasterLib_v1.csv.gz", index=False, compression="gzip"
)

master.drop(columns=["sgRNA", "Gene"]).to_csv(
    f"{RPATH}/MasterLib_v1.xlsx", index=False, compression="gzip"
)
