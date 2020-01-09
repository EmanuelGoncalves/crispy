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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from crispy.CrispyPlot import CrispyPlot
from crispy.Enrichment import Enrichment
from crispy.DataImporter import Proteomics
from scipy.stats import normaltest, shapiro
from statsmodels.stats.multitest import multipletests
from notebooks.swath_proteomics.SLethal import SLethal


# Proteomics data-set

prot = Proteomics()

prot_all = prot.get_data()
prot_flt = prot.get_data(dtype="noimputed")


# Gene-sets

gset = Enrichment(
    gmts=["c5.all.v6.2.symbols.gmt"], sig_min_len=5, padj_method="fdr_bh", verbose=1
)


# Hypergeometric test

values = prot_flt.median(1)
background = {g for gs, s in gset.gmts["c5.all.v6.2.symbols.gmt"].items() for g in s}
sublist = set(prot_all.index).intersection(background)

measured_enr = gset.hypergeom_enrichments(sublist, background, "c5.all.v6.2.symbols.gmt")
abundance_enr = gset.gsea_enrichments(values, "c5.all.v6.2.symbols.gmt")
