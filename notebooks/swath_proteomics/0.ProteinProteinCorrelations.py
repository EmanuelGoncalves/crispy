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
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
from crispy.DataImporter import Proteomics, CORUM


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# GDSC baseline cancer cell lines SWATH-Proteomics analysis
#

data = Proteomics().filter(perc_measures=.75)
data = data.drop(columns=["HEK", "h002"])
data = data.T.fillna(data.T.mean()).T
LOG.info(f"Proteomics: {data.shape}")


# Protein-Protein LMMs
#

lmm = LMModels()

# Covariates
m = lmm.define_covariates()

# Matrices
y, x = data.copy().T, data.copy().T

# Kinship matrix (random effects)
k = lmm.kinship(x)

# Protein associations
res = lmm.matrix_limix_lmm(Y=y, X=x, M=m, K=k)

# Protein complexes (CORUM)
corum = CORUM().db_melt_symbol
res = res.assign(corum=[int((p1, p2) in corum) for p1, p2 in res[["y_id", "x_id"]].values])

# Export
res.to_csv(f"{RPATH}/lmm_protein_protein.csv", index=False, compression="gzip")


# CORUM
#

_, ax = plt.subplots(1, 1, figsize=(2., 2.), dpi=600)

for i, df in res.groupby("corum"):
    sns.distplot(
        df["beta"],
        hist_kws=dict(alpha=0.4, zorder=1, linewidth=0),
        bins=30,
        kde_kws=dict(cut=0, lw=1, zorder=1, alpha=0.8),
        label="CORUM" if i else "Rest",
        color=CrispyPlot.PAL_DBGD[i],
        ax=ax,
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.set_xlabel("Association beta")
ax.set_ylabel("Density")

ax.legend(prop={"size": 6}, loc=2, frameon=False)

plt.savefig(
    f"{RPATH}/lmm_protein_protein_CORUM_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
