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
from scipy import stats
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.GIPlot import GIPlot
from swath_proteomics.LMModels import LMModels
from sklearn.metrics import roc_auc_score, roc_curve
from crispy.DataImporter import Proteomics, CORUM, GeneExpression


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


def gkn(values):
    kernel = stats.gaussian_kde(values.dropna())
    return {
        k: np.log(kernel.integrate_box_1d(-1e8, v) / kernel.integrate_box_1d(v, 1e8))
        if not np.isnan(v)
        else np.nan
        for k, v in values.to_dict().items()
    }


# Protein abundance attenuation
#

p_attenuated = pd.read_csv(f"{DPATH}/protein_attenuation_table.csv", index_col=0)


# SWATH-Proteomics
#

prot = Proteomics().filter(perc_measures=0.75).drop(columns=["HEK", "h002"])
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


# Normalise
#

prot_norm = pd.DataFrame({g: gkn(prot.loc[g, samples]) for g in genes}).T
gexp_norm = pd.DataFrame({g: gkn(gexp.loc[g, samples]) for g in genes}).T


# Protein-Protein LMMs
#

lmm = LMModels(y=prot_norm.T, x=gexp_norm.T, institute=False, x_feature_type="same_y")

# Protein associations
res = lmm.matrix_lmm(pval_adj_overall=True)

# Protein attenuation
res["attenuation"] = p_attenuated.reindex(res["y_id"])["attenuation_potential"].replace(np.nan, "NA").values

# Export
res.to_csv(f"{RPATH}/corr_protein_gexp.csv", index=False, compression="gzip")


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
    f"{RPATH}/corr_protein_gexp_histogram.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Protein attenuation
#

pal = dict(zip(*(["Low", "High", "NA"], CrispyPlot.PAL_DBGD.values())))

_, ax = plt.subplots(1, 1, figsize=(1.0, 1.5), dpi=600)
sns.boxplot(
    x="attenuation",
    y="beta",
    notch=True,
    data=res,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=pal,
    showcaps=False,
    saturation=1,
    ax=ax,
)
ax.set_xlabel("Attenuation")
ax.set_ylabel("Transcript ~ Protein")
ax.axhline(0, ls="-", lw=0.1, c="black", zorder=0)
ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
plt.savefig(f"{RPATH}/corr_protein_gexp_attenuation_boxplot.pdf", bbox_inches="tight")
plt.close("all")


# Representative examples
#

for gene in ["VIM", "EGFR", "IDH2", "NRAS", "SMARCB1", "ERBB2", "STAG1", "STAG2", "TP53"]:
    plot_df = pd.concat(
        [
            prot.loc[gene, samples].rename("protein"),
            gexp.loc[gene, samples].rename("transcript"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_regression("protein", "transcript", plot_df)
    grid.set_axis_labels(f"Protein", f"Transcript")
    grid.ax_marg_x.set_title(gene)
    plt.savefig(f"{RPATH}/corr_protein_gexp_scatter_{gene}.pdf", bbox_inches="tight")
    plt.close("all")
