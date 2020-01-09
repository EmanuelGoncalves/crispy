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
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
from sklearn.metrics import roc_auc_score, roc_curve
from crispy.DataImporter import Proteomics, CORUM, GeneExpression


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH-Proteomics
#

prot = Proteomics().filter(perc_measures=0.75)
prot = prot.drop(columns=["HEK", "h002"])
LOG.info(f"Proteomics: {prot.shape}")


# Gene-sets
#

gset = Enrichment(
    gmts=["c5.bp.v6.2.symbols.gmt"], sig_min_len=5, padj_method="fdr_bh", verbose=1
)


# Hypergeometric test
#

background = {g for gs, s in gset.gmts["c5.all.v6.2.symbols.gmt"].items() for g in s}
sublist = set(prot.index).intersection(background)
LOG.info(f"Gene: {len(sublist)}")

measured_enr = gset.hypergeom_enrichments(sublist, background, "c5.bp.v6.2.symbols.gmt")
measured_enr.to_csv(f"{RPATH}/protein_detection_enrichment.csv", index=False, compression="gzip")
LOG.info(measured_enr[measured_enr["adj.p_value"] < .01])