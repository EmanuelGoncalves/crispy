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
import sys
import crispy
import logging
import argparse
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from natsort import natsorted
from itertools import zip_longest
from sklearn.mixture import GaussianMixture
from swath_proteomics.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR, CopyNumber


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Data-sets
#
prot, gexp, cnv = Proteomics(), GeneExpression(), CopyNumber()


# Samples
#
samples = set.intersection(
    set(prot.get_data()), set(gexp.get_data()), set(cnv.get_data())
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#
prot = prot.filter(subset=samples, perc_measures=0.75)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
LOG.info(f"Transcriptomics: {gexp.shape}")

cnv = cnv.filter(subset=samples)
LOG.info(f"Copy number: {cnv.shape}")


# Genes
#
genes = natsorted(
    list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
)
LOG.info(f"Genes: {len(genes)}")


# Import regressions with CNV
#
cnv_prot = pd.read_csv(f"{RPATH}/lmm_protein_cnv.csv.gz", index_col=0)
cnv_gexp = pd.read_csv(f"{RPATH}/lmm_gexp_cnv.csv.gz", index_col=0)
cnv_prot_gexp = pd.read_csv(f"{RPATH}/lmm_protein_cnv_gexp.csv.gz")


# Assemble attenuation data-frame
#

patt = pd.concat(
    [cnv_prot["beta"].rename("protein"), cnv_gexp["beta"].rename("gexp")],
    axis=1,
    sort=False,
)

patt = patt.assign(att=patt.eval("gexp - protein").values)
patt = patt.sort_values("att", ascending=False)

# GMM
gmm = GaussianMixture(n_components=2).fit(patt[["att"]])
s_type, clusters = (
    pd.Series(gmm.predict(patt[["att"]]), index=patt.index),
    pd.Series(gmm.means_[:, 0], index=range(2)),
)
patt["cluster"] = [
    "High" if s_type[i] == clusters.argmax() else "Low" for i in patt.index
]


# Attenuation scatter
#

ax_min = patt[["protein", "gexp"]].min().min() * 1.1
ax_max = patt[["protein", "gexp"]].max().max() * 1.1

pal = dict(High="#E74C3C", Low="#99A3A4")

g = sns.jointplot(
    "gexp",
    "protein",
    patt,
    "scatter",
    color="#808080",
    xlim=[ax_min, ax_max],
    ylim=[ax_min, ax_max],
    space=0,
    s=5,
    edgecolor="w",
    linewidth=0.0,
    marginal_kws={"hist": False, "rug": False},
    stat_func=None,
    alpha=0.1,
)

for n, df in patt.groupby("cluster"):
    g.x, g.y = df["gexp"], df["protein"]
    g.plot_joint(
        sns.regplot,
        color=pal[n],
        fit_reg=False,
        scatter_kws={"s": 5, "alpha": 0.5, "linewidth": 0},
    )
    g.plot_joint(
        sns.kdeplot,
        cmap=sns.light_palette(pal[n], as_cmap=True),
        legend=False,
        shade=False,
        shade_lowest=False,
        n_levels=9,
        alpha=0.8,
        lw=0.1,
    )
    g.plot_marginals(sns.kdeplot, color=pal[n], shade=True, legend=False)

handles = [mpatches.Circle([0, 0], .25, facecolor=pal[s], label=s) for s in pal]
g.ax_joint.legend(loc='upper left', handles=handles, title='Protein\nattenuation', frameon=False)

g.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
g.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)

plt.gcf().set_size_inches(2.5, 2.5)
g.set_axis_labels("GExp ~ CopyNumber (beta)", "Protein ~ CopyNumber (beta)")
plt.savefig(f"{RPATH}/lmm_cnv_scatter.pdf", bbox_inches="tight")
plt.close("all")
