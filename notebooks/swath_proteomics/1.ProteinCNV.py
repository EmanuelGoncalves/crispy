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
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, CopyNumber


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

cnv_prot = LMModels(
    y=prot.T[genes], x=cnv.T[genes], institute=False, transform_x="None", x_feature_type="same_y"
).matrix_lmm().sort_values("fdr")
cnv_prot.to_csv(f"{RPATH}/lmm_protein_cnv.csv.gz", index=False, compression="gzip")

cnv_gexp = LMModels(
    y=gexp.T[genes], x=cnv.T[genes], institute=False, transform_x="None", x_feature_type="same_y"
).matrix_lmm().sort_values("fdr")
cnv_gexp.to_csv(f"{RPATH}/lmm_gexp_cnv.csv.gz", index=False, compression="gzip")


# Regressions Prot ~ CNV + GExp
#

cnv_prot_gexp = LMModels(
    y=prot.T[genes], x=cnv.T[genes], m2=gexp.T[genes], institute=False, transform_x="None", x_feature_type="same_y"
).matrix_lmm().sort_values("fdr")
cnv_prot_gexp.to_csv(f"{RPATH}/lmm_protein_cnv_gexp.csv.gz", index=False, compression="gzip")
