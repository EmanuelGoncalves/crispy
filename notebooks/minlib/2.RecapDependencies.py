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
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from minlib import dpath, rpath, LOG
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.CRISPRData import CRISPRDataSet
from sklearn.metrics import mean_squared_error
from crispy.DataImporter import PipelineResults


# Manifest

manifest = pd.read_csv(f"{dpath}/minimal_lib/manifest.csv", index_col=0)
manifest["lib"] = [";".join(set(i.split(", "))) for i in manifest["sgRNA library"]]


# CRISPR pipeline results

cres_min = PipelineResults(
    f"{dpath}/minimal_lib/minlib_dep/",
    import_fc=True,
    import_bagel=True,
    import_mageck=True,
)

cres_all = PipelineResults(
    f"{dpath}/minimal_lib/KYlib_dep/",
    import_fc=True,
    import_bagel=True,
    import_mageck=True,
)


# Overlap

genes = list(set(cres_min.bf_b.index).intersection(cres_all.bf_b.index))
samples = list(set(cres_min.bf_b).intersection(cres_all.bf_b))
LOG.info(f"Genes={len(genes)}; Samples={len(samples)}")


#

plot_df = pd.concat(
    [
        cres_all.bf_b.loc[genes].sum(1).rename("all"),
        cres_min.bf_b.loc[genes].sum(1).rename("min"),
    ],
    axis=1,
    sort=False,
)

plt.scatter(plot_df["all"], plot_df["min"])
plt.show()


# Correlate number of dependencies found per cell lines between small and full library

cres_min.bf_b
cres_all.bf_b


# Agreement difference between high and low quality samples
