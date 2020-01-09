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


# GeneExpression
#

gexp = GeneExpression().filter()
LOG.info(f"Transcriptomics: {gexp.shape}")


# Overlaps
#

samples = list(set(prot).intersection(gexp))
genes = list(set(prot.index).intersection(gexp.index))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Protein-Protein LMMs
#

lmm = LMModels()

# Covariates
m = lmm.define_covariates()

# Matrices
y, x = prot.copy().T, gexp.copy().T

# Kinship matrix (random effects)
k = lmm.kinship(x)

# Protein associations
res = lmm.matrix_limix_lmm(Y=y, X=x, M=m, K=k, x_feature_type="same_y", pval_adj_overall=True)

# Export
res.to_csv(f"{RPATH}/lmm_protein_gexp.csv", index=False, compression="gzip")


# Protein ~ GExp correlation histogram
#

_, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=600)

sns.distplot(
    res["beta"],
    hist_kws=dict(alpha=0.4, zorder=1, linewidth=0),
    bins=30,
    kde_kws=dict(cut=0, lw=1, zorder=1, alpha=0.8),
    color=CrispyPlot.PAL_DBGD[0],
    ax=ax,
)

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.set_xlabel("Association beta")
ax.set_ylabel("Density")

plt.savefig(
    f"{RPATH}/lmm_protein_gexp_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
