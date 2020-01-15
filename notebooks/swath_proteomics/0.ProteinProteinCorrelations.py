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
from crispy.DataImporter import Proteomics, CORUM
from sklearn.metrics import roc_auc_score, roc_curve


LOG = logging.getLogger("Crispy")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH-Proteomics
#

data = Proteomics().filter()
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
res = lmm.matrix_limix_lmm(Y=y, X=x, M=m, K=k, x_feature_type="drop_y")

# Protein complexes (CORUM)
corum = set(CORUM().db_melt_symbol)
res = res.assign(
    corum=[int((p1, p2) in corum) for p1, p2 in res[["y_id", "x_id"]].values]
)

# Export
res.to_csv(f"{RPATH}/lmm_protein_protein.csv", index=False, compression="gzip")


# CORUM - Histogram
#

_, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=600)

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

ax.legend(prop={"size": 6}, frameon=False)

plt.savefig(
    f"{RPATH}/lmm_protein_protein_CORUM_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# CORUM - ROC curve
#

x, y, xy_auc = QCplot.recall_curve(
    -res["beta"], index_set=set(res.query("corum == 1").index)
)

_, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=600)

ax.plot(x, y, label=f"CORUM (AUC={xy_auc:.2f})", color=CrispyPlot.PAL_DBGD[0])
ax.plot([1, 0], [1, 0], "k-", lw=0.3, zorder=0)
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.set_xlabel("Precent-rank of associations effect size")
ax.set_ylabel("Cumulative fraction")

ax.legend(prop={"size": 5}, frameon=False)

plt.savefig(
    f"{RPATH}/lmm_protein_protein_CORUM_recall_curve.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Protein-protein associations volcano
#

_, ax = plt.subplots(1, 1, figsize=(4.0, 2.0), dpi=600)

for i, df in res.query("fdr < .05").groupby("corum"):
    ax.scatter(
        -np.log10(df["pval"]),
        df["beta"],
        s=3,
        color=CrispyPlot.PAL_DBGD[i],
        label="CORUM" if i else "Rest",
        edgecolor="white",
        lw=0.1,
        alpha=0.5,
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.set_ylabel("Effect size (beta)")
ax.set_xlabel("Association p-value (-log10)")

ax.legend(prop={"size": 5}, frameon=False)

plt.savefig(
    f"{RPATH}/lmm_protein_protein_volcano.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")
