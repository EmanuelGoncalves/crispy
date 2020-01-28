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
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from crispy.GIPlot import GIPlot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import Proteomics, GeneExpression, CopyNumber


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH proteomics
#

prot = Proteomics().filter(perc_measures=None)
LOG.info(f"Proteomics: {prot.shape}")


# Gene expression
#

gexp = GeneExpression().filter(subset=list(prot))
LOG.info(f"Transcriptomics: {gexp.shape}")


# Copy number
#

cnv = CopyNumber().filter(subset=list(gexp))
LOG.info(f"Copy number: {cnv.shape}")


# Overlaps
#

samples = list(set.intersection(set(prot), set(gexp), set(cnv)))
genes = list(set.intersection(set(prot.index), set(gexp.index), set(cnv.index)))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Correlations with copy number
#


def corr_genes(g):
    m = pd.concat(
        [
            prot.loc[g].rename("proteomics"),
            gexp.loc[g].rename("transcriptomics"),
            cnv.loc[g].rename("copynumber"),
        ],
        axis=1,
        sort=False,
    ).dropna()
    prot_r, prot_p = pearsonr(m["copynumber"], m["proteomics"])
    gexp_r, gexp_p = pearsonr(m["copynumber"], m["transcriptomics"])
    return dict(
        gene=g,
        prot_r=prot_r,
        prot_p=prot_p,
        gexp_r=gexp_r,
        gexp_p=gexp_p,
        len=m.shape[0],
    )


patt = pd.DataFrame([corr_genes(g) for g in genes]).query(f"len > {0.5 * len(samples)}")
patt = patt.assign(attenuation=patt.eval("gexp_r - prot_r"))

# GMM
gmm = GaussianMixture(n_components=2).fit(patt[["attenuation"]])
s_type, clusters = (
    pd.Series(gmm.predict(patt[["attenuation"]]), index=patt.index),
    pd.Series(gmm.means_[:, 0], index=range(2)),
)
patt["cluster"] = [
    "High" if s_type[i] == clusters.argmax() else "Low" for i in patt.index
]


# Attenuation scatter
#

ax_min = patt[["prot_r", "gexp_r"]].min().min() * 1.1
ax_max = patt[["prot_r", "gexp_r"]].max().max() * 1.1

pal = dict(High=CrispyPlot.PAL_DTRACE[1], Low=CrispyPlot.PAL_DTRACE[0])

g = sns.jointplot(
    "gexp_r",
    "prot_r",
    patt,
    "scatter",
    color=CrispyPlot.PAL_DTRACE[0],
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

for n in ["Low", "High"]:
    df = patt.query(f"cluster == '{n}'")
    g.x, g.y = df["gexp_r"], df["prot_r"]
    g.plot_joint(
        sns.regplot,
        color=pal[n],
        fit_reg=False,
        scatter_kws={"s": 3, "alpha": 0.5, "linewidth": 0},
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

handles = [mpatches.Circle([0, 0], 0.25, facecolor=pal[s], label=s) for s in pal]
g.ax_joint.legend(
    loc="upper left", handles=handles, title="Protein\nattenuation", frameon=False
)

g.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
g.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)

plt.gcf().set_size_inches(2.5, 2.5)
g.set_axis_labels(
    "Transcriptomics ~ Copy number\n(Pearson's R)",
    "Protein ~ Copy number\n(Pearson's R)",
)
plt.savefig(f"{RPATH}/1.ProteinAttenuation_scatter.pdf", bbox_inches="tight")
plt.close("all")


# Pathway enrichement analysis of attenuated proteins
#

background = set(patt["gene"])
sublist = set(patt.query("cluster == 'High'")["gene"])

plot_df = Enrichment(
    gmts=["c5.cc.v7.0.symbols.gmt"], sig_min_len=15, padj_method="fdr_bh"
).hypergeom_enrichments(sublist, background, "c5.cc.v7.0.symbols.gmt")
plot_df = plot_df[plot_df["adj.p_value"] < 0.01].head(30).reset_index()
plot_df["gset"] = [i[3:].lower().replace("_", " ") for i in plot_df["gset"]]

_, ax = plt.subplots(1, 1, figsize=(2.0, 5.0), dpi=600)

sns.barplot(
    -np.log10(plot_df["adj.p_value"]),
    plot_df["gset"],
    orient="h",
    color=CrispyPlot.PAL_DTRACE[2],
    ax=ax,
)

for i, (_, row) in enumerate(plot_df.iterrows()):
    plt.text(
        -np.log10(row["adj.p_value"]),
        i,
        f"{row['len_intersection']}/{row['len_sig']}",
        va="center",
        ha="left",
        fontsize=5,
        zorder=10,
        color=CrispyPlot.PAL_DTRACE[2],
    )

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.set_xlabel("Hypergeometric test (-log10 FDR)")
ax.set_ylabel("")
ax.set_title(f"GO celluar component enrichment - attenuated proteins")

plt.savefig(
    f"{RPATH}/1.ProteinAttenuation_enrichment.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")
