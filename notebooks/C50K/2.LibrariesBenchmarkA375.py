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

import datetime
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from crispy.QCPlot import QCplot
from crispy.Utils import Utils
from scipy.interpolate import interpn
from sklearn.metrics import mean_squared_error
from crispy.CopyNumberCorrection import Crispy
from scipy.stats import spearmanr, mannwhitneyu
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from crispy.DataImporter import Mobem, CopyNumber, CopyNumberSegmentation
from C50K import (
    clib_palette,
    clib_order,
    define_sgrnas_sets,
    estimate_ks,
    sgrnas_scores_scatter,
    guides_aroc_benchmark,
    ky_v11_calculate_gene_fc,
    lm_associations,
    project_score_sample_map,
    assemble_crisprcleanr,
    assemble_crispy,
)


# -
dpath = pkg_resources.resource_filename("crispy", "data/")
rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

