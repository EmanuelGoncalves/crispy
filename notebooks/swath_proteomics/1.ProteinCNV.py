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

import os
import sys
import logging
import argparse
import pandas as pd
import pkg_resources
from natsort import natsorted
from itertools import zip_longest
from swath_proteomics.LMModels import LMModels
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, CopyNumber, CORUM


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Data-sets
#
prot, gexp, cnv = Proteomics(), GeneExpression(), CopyNumber()


# Samples
#
samples = set.intersection(
    set(prot.get_data()), set(gexp.get_data()), set(cnv.get_data())
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#
prot = prot.filter(subset=samples, perc_measures=0.75)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
LOG.info(f"Transcriptomics: {gexp.shape}")

cnv = cnv.filter(subset=samples)
LOG.info(f"Copy number: {cnv.shape}")


# Genes
#
genes = natsorted(
    list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
)
LOG.info(f"Genes: {len(genes)}")


# Regressions with copy number
#

cnv_prot = (
    LMModels(
        y=prot.T[genes],
        x=cnv.T[genes],
        institute=False,
        transform_x="None",
        x_feature_type="same_y",
    )
    .matrix_lmm()
    .sort_values("fdr")
)
cnv_prot.to_csv(f"{RPATH}/lmm_protein_cnv.csv.gz", index=False, compression="gzip")

cnv_gexp = (
    LMModels(
        y=gexp.T[genes],
        x=cnv.T[genes],
        institute=False,
        transform_x="None",
        x_feature_type="same_y",
    )
    .matrix_lmm()
    .sort_values("fdr")
)
cnv_gexp.to_csv(f"{RPATH}/lmm_gexp_cnv.csv.gz", index=False, compression="gzip")


# CORUM regressions Prot ~ CNV + GExp
#

corum = CORUM().db_melt_symbol
corum = pd.DataFrame(
    [dict(p1=p1, p2=p2) for p1, p2 in corum if p1 in genes and p2 in genes]
)

corum_res = []
for p1, p1_df in corum.groupby("p1"):
    LOG.info(f"CORUM pairs: p1 = {p1}, p2s = {len(p1_df)}")
    p1_df_res = (
        LMModels(
            y=prot.T[[p1]],
            x=cnv.T[p1_df["p2"]],
            m2=gexp.T[[p1]],
            institute=False,
            transform_x="None",
        )
        .matrix_lmm(pval_adj_overall=True)
        .sort_values("fdr")
    )
    corum_res.append(p1_df_res)
corum_res = pd.concat(corum_res, ignore_index=True)
corum_res["fdr"] = multipletests(corum_res["pval"], method="fdr_bh")[1]
corum_res = corum_res.sort_values("fdr")
corum_res.to_csv(f"{RPATH}/lmm_protein_corum.csv.gz", index=False, compression="gzip")
