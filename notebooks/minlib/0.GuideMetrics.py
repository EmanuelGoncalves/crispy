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
from crispy.CRISPRData import CRISPRDataSet
from minlib import dpath, rpath, define_sgrnas_sets, estimate_ks


# Project Score samples acquired with Kosuke_Yusa v1.1 library

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_count.counts.remove_low_counts(ky_v11_count.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_count.plasmids)
)


# Define sets of sgRNAs

sgrna_sets = define_sgrnas_sets(ky_v11_count.lib, ky_v11_fc, add_controls=True)


# Test sgRNA distribution difference to non-targetting guides (Kolmogorov-Smirnov)

ky_v11_ks = estimate_ks(ky_v11_fc, sgrna_sets["nontargeting"]["fc"])
ky_v11_ks["ks_control_min"] = 1 - ky_v11_ks["ks_control"]


# JACKS guide efficacy scores

jacks = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0.csv", index_col=0)
ky_v11_ks["jacks"] = jacks.reindex(ky_v11_ks.index)["X1"].values
ky_v11_ks["jacks_min"] = abs(ky_v11_ks["jacks"] - 1)


# Doench-Root efficacy scores

doenchroot = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0_DoenchRoot.csv", index_col=0)
ky_v11_ks["doenchroot"] = doenchroot.reindex(ky_v11_ks.index)["score_with_ppi"].values
ky_v11_ks["doenchroot_min"] = 1 - ky_v11_ks["doenchroot"]


# FORECAST In Frame Percentage

forecast = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0_ForecastFrameshift.csv", index_col=0)
ky_v11_ks["forecast"] = forecast.reindex(ky_v11_ks.index)["In Frame Percentage"].values


# Gene target annotation

ky_v11_ks["Gene"] = ky_v11_count.lib.reindex(ky_v11_ks.index)["Gene"].values


# Export

ky_v11_ks.to_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx")

