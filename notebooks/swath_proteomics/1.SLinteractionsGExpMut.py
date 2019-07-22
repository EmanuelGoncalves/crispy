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
from scipy.stats import normaltest, shapiro
from statsmodels.stats.multitest import multipletests
from notebooks.swath_proteomics.SLethal import SLethal


logger = logging.getLogger("Crispy")
rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Import data-sets

slethal = SLethal(
    use_proteomics=False,
    filter_gexp_kws=dict(dtype="voom", normality=False),
    filter_crispr_kws=dict(scale=True, abs_thres=0.5, iqr_range=(10, 90), iqr_range_thres=0.5),
    filter_wes_kws=dict(
        as_matrix=True, mutation_class=["frameshift", "nonsense", "ess_splice"]
    ),
)


# GI

y = slethal.crispr[slethal.samples]


# Gene-expression GI

x = slethal.gexp[slethal.samples]

gi_gexp = slethal.genetic_interactions(y=y, x=x)
gi_gexp.to_csv(f"{rpath}/gis_gexp.csv.gz", compression="gzip")


# Copy-number GI

x = slethal.cn[slethal.samples]
x = x.loc[x.std(1) != 0]

gi_cn = slethal.genetic_interactions(y=y, x=x)
gi_cn.to_csv(f"{rpath}/gis_cn.csv.gz", compression="gzip")


# WES GI

x = slethal.wes[slethal.samples]
x = x.loc[x.std(1) != 0]

gi_wes = slethal.genetic_interactions(y=y, x=x)
gi_wes.to_csv(f"{rpath}/gis_wes.csv.gz", compression="gzip")


# -
genes = ["ARID2", "PBRM1", "ARID1A", "ARID1B", "TP53"]

gi_subset_wes = pd.concat([gi_wes.loc[g].head(15).assign(phenotype=g) for g in genes if g in slethal.crispr.index])
gi_subset_wes = gi_subset_wes.sort_values(["adjpval", "pval"])
gi_subset_wes.to_clipboard()

gi_subset_gexp = pd.concat([gi_gexp.loc[g].head(15).assign(phenotype=g) for g in genes if g in slethal.crispr.index])
gi_subset_gexp = gi_subset_gexp.sort_values(["adjpval", "pval"])
gi_subset_gexp.to_clipboard()
