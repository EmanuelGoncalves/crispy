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
import logging
import matplotlib
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from scipy.stats import spearmanr
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
from depmap.GExpThres import dim_reduction, plot_dim_reduction
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    Methylation,
    Sample
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Data-sets
#

prot_obj = Proteomics()
methy_obj = Methylation()
gexp_obj = GeneExpression()


# Samples
#

ss = Sample().samplesheet

samples = set.intersection(
    set(prot_obj.get_data()), set(gexp_obj.get_data()), set(methy_obj.get_data())
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#

prot = prot_obj.filter(subset=samples, perc_measures=0.75)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp_obj.filter(subset=samples)
gexp = gexp.loc[gexp.std(1) > 1.5]
LOG.info(f"Transcriptomics: {gexp.shape}")

methy = methy_obj.filter(subset=samples)
methy = methy.loc[methy.std(1) > 0.1]
LOG.info(f"Methylation: {methy.shape}")


# MOFA
#

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy),
    iterations=2000,
    convergence_mode="fast",
    factors_n=100,
    from_file=f"{RPATH}/1.MultiOmicsDimRed.hdf5",
)
mofa.save_hdf5(f"{RPATH}/1.MultiOmicsDimRed.hdf5")


# Dimension reduction
#

factors_tsne, factors_pca = dim_reduction(mofa.factors.T, pca_ncomps=50, input_pca_to_tsne=False)

dimred = dict(tSNE=factors_tsne, pca=factors_pca)

for ctype, df in dimred.items():
    plot_df = pd.concat(
        [df, ss["tissue"]], axis=1, sort=False
    ).dropna()

    ax = plot_dim_reduction(plot_df, ctype=ctype, palette=CrispyPlot.PAL_TISSUE_2)
    ax.set_title(f"Factors {ctype}")
    plt.savefig(
        f"{RPATH}/1.MultiOmicsDimRed_factors_{ctype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
