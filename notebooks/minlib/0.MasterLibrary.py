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
import pandas as pd
import pkg_resources
from crispy.Utils import Utils
from scipy.stats import ks_2samp
from crispy.CRISPRData import CRISPRDataSet, Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


def estimate_ks(fc, control_guides_fc, verbose=0):
    ks_sgrna = []
    for sgrna in fc.index:
        if verbose > 0:
            LOG.info(f"sgRNA={sgrna}")

        d_ctrl, p_ctrl = ks_2samp(fc.loc[sgrna], control_guides_fc)

        ks_sgrna.append({"sgrna": sgrna, "ks_control": d_ctrl, "pval_control": p_ctrl})

    ks_sgrna = pd.DataFrame(ks_sgrna).set_index("sgrna")
    ks_sgrna["median_fc"] = fc.loc[ks_sgrna.index].median(1).values

    return ks_sgrna


def define_sgrnas_sets(clib, fc=None, add_controls=True, dataset_name="Yusa_v1"):
    sgrna_sets = dict()

    # sgRNA essential
    sgrnas_essential = Utils.get_essential_genes(return_series=False)
    sgrnas_essential = set(clib[clib["Gene"].isin(sgrnas_essential)].index)
    sgrnas_essential_fc = (
        None if fc is None else fc.reindex(sgrnas_essential).median(1).dropna()
    )

    sgrna_sets["essential"] = dict(
        color="#e6550d", sgrnas=sgrnas_essential, fc=sgrnas_essential_fc
    )

    # sgRNA non-essential
    sgrnas_nonessential = Utils.get_non_essential_genes(return_series=False)
    sgrnas_nonessential = set(clib[clib["Gene"].isin(sgrnas_nonessential)].index)
    sgrnas_nonessential_fc = (
        None if fc is None else fc.reindex(sgrnas_nonessential).median(1).dropna()
    )

    sgrna_sets["nonessential"] = dict(
        color="#3182bd", sgrnas=sgrnas_nonessential, fc=sgrnas_nonessential_fc
    )

    # sgRNA non-targeting
    if add_controls:
        if dataset_name in ["Yusa_v1", "Yusa_v1.1", "Sabatini_Lander_AML"]:
            sgrnas_control = {i for i in clib.index if i.startswith("CTRL0")}
        else:
            sgrnas_control = set(
                clib[[i.startswith("NO_CURRENT_") for i in clib["Gene"]]].index
            )

        sgrnas_control_fc = fc.reindex(sgrnas_control).median(1).dropna()

        sgrna_sets["nontargeting"] = dict(
            color="#31a354",
            sgrnas=sgrnas_control,
            fc=None if fc is None else sgrnas_control_fc,
        )

    return sgrna_sets


# Project Score KY v1.1
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_fc = ky.counts.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)
ky_gsets = define_sgrnas_sets(ky.lib, ky_fc, add_controls=True)
ky_ks = estimate_ks(ky_fc, ky_gsets["nontargeting"]["fc"])


# DepMap Q2 Avana
#

avana = CRISPRDataSet("Avana")
avana_fc = (
    avana.counts.remove_low_counts(avana.plasmids).norm_rpm().foldchange(avana.plasmids)
)
avana_gsets = define_sgrnas_sets(
    avana.lib, avana_fc, dataset_name="Avana", add_controls=True
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


# KS scores
#

ks = pd.concat([ky_ks["ks_control"], avana_ks["ks_control"]])


# JACKS scores
#

jacks_v1 = pd.read_csv(f"{DPATH}/jacks/Yusa_v1.0.csv", index_col=0)["X1"]
jacks_v11 = pd.read_csv(f"{DPATH}/jacks/Yusa_v1.1.csv", index_col=0)["X1"]
jacks_ky = pd.concat([jacks_v1, jacks_v11], axis=1, sort=False).min(1)

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

master = pd.read_csv(f"{DPATH}/update/master.csv.gz").drop(
    columns=["doenchroot", "jacks_min", "ks_control_min"]
)
master.loc[master["sgRNA_ID"].isin(ky_v11_wge.index), "sgRNA"] = (
    ky_v11_wge.loc[master["sgRNA_ID"], "WGE_sequence"].dropna().values
)

master["Approved symbol"] = hgnc.reindex(master["Gene"])["Approved symbol"].values
master["Approved name"] = hgnc.reindex(master["Gene"])["Approved name"].values
master["HGNC ID"] = hgnc.reindex(master["Gene"])["HGNC ID"].values

master["WGE_Sequence"] = wge.reindex(master["sgRNA"])["WGE_Sequence"].values
master["WGE_ID"] = wge.reindex(master["sgRNA"])["WGE"].values
master["Off_Target"] = wge.reindex(master["sgRNA"])["OT_Summary"].values
master["WGE_MatchComment"] = wge.reindex(master["sgRNA"])["Comments"].values

master["JACKS"] = jacks.reindex(master["sgRNA_ID"]).values
master["RuleSet2"] = ruleset2.reindex(master["sgRNA_ID"]).values
master["FORECAST"] = forecast_ky.reindex(master["sgRNA_ID"]).values
master["KS"] = ks.reindex(master["sgRNA_ID"]).values

master.sort_values(
    ["lib", "Approved symbol", "KS"], ascending=[True, True, False]
).drop(columns=["sgRNA", "Gene"]).to_csv(
    f"{DPATH}/crispr_libs/MasterLib_v1.csv.gz", index=False, compression="gzip"
)
