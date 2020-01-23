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
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
from crispy.DataImporter import (
    Proteomics,
    GeneExpression,
    Methylation,
    CRISPR,
    Sample,
    WES,
)


FDR_THRES = 0.05
LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Data-sets
#
prot, gexp, crispr, methy, wes = (
    Proteomics(),
    GeneExpression(),
    CRISPR(),
    Methylation(),
    WES(),
)


# Samples
#
ss = Sample().samplesheet

samples = set.intersection(
    set(prot.get_data()),
    set(gexp.get_data()),
    set(methy.get_data()),
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#
prot = prot.filter(subset=samples, perc_measures=0.75, replicate_thres=0.7)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
gexp = gexp.loc[gexp.std(1) > 1.5]
LOG.info(f"Transcriptomics: {gexp.shape}")

methy = methy.filter(subset=samples)
methy = methy.loc[methy.std(1) > 0.1]
LOG.info(f"Methylation: {methy.shape}")

wes = wes.filter(subset=samples, min_events=3, recurrence=True)
LOG.info(f"WES: {wes.shape}")

crispr = crispr.filter(subset=samples, abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")


# Covariates
#

covariates = pd.concat(
    [
        Proteomics().rep_corrs.groupby("SIDM_1")["pearson"].mean().rename("reps_corr").loc[samples],
        prot.count().loc[samples].rename("n_measurements"),
        pd.get_dummies(ss["growth_properties"]).loc[samples],
        (ss["tissue"] == "Haematopoietic and Lymphoid").astype(int).loc[samples].rename("haem"),
    ],
    axis=1,
).dropna()


# MOFA
#

mofa = MOFA(
    views=dict(proteomics=prot.T, transcriptomics=gexp.T, methylation=methy.T),
    covariates=covariates,
    iterations=1000,
    scale_views=True,
    convergence_mode="slow",
    likelihoods=["gaussian", "gaussian", "gaussian"],
    factors_n=35,
)


# Factor association analysis
#

lmm_crispr = LMModels(y=mofa.factors, x=crispr.T)
lmm_crispr_df = lmm_crispr.matrix_lmm()
print(lmm_crispr_df.query("fdr < 0.05").head(60))


# Variance explained across data-sets
#

MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/2.rsquared_heatmap.pdf", bbox_inches="tight")
plt.close("all")


# Factors data-frame integrated with other measurements
#

plot_df = pd.concat(
    [
        mofa.factors,
        mofa.views["transcriptomics"][["CDH1"]].add_suffix("_transcriptomics"),
        mofa.views["methylation"][["PTPRT"]].add_suffix("_methylation"),
        mofa.views["proteomics"][["VIM"]].add_suffix("_proteomics"),
        crispr.loc[["HNRNPA1L2", "FH", "ELP4", "ELOVL1"]].T.add_suffix("_crispr"),
        covariates,
    ],
    axis=1,
    sort=False,
).dropna()
print(plot_df.corr()["F1"].sort_values())


# Regression
#

g, f = "ELOVL1_crispr", "F19"
grid = GIPlot.gi_regression(g, f, plot_df)
grid.set_axis_labels(f"{g}\nCRISPR (scaled FC)", f"Factor {f[1:]}")
plt.savefig(f"{RPATH}/2.regression_{f}_{g}.pdf", bbox_inches="tight")
plt.close("all")


# Top features
#

for factor in ["F1"]:
    for view in mofa.views:
        # Weights scatter
        MOFAPlot.factor_weights_scatter(mofa, view, factor)
        plt.savefig(f"{RPATH}/2.top_features_{factor}_{view}.pdf", bbox_inches="tight")
        plt.close("all")

        # Data heatmaps
        MOFAPlot.view_heatmap(
            mofa,
            view,
            factor,
            col_colors=CrispyPlot.get_palettes(samples, ss),
            title=f"{view} heatmap of {factor} top features",
        )
        plt.savefig(
            f"{RPATH}/2.top_features_{factor}_{view}_heatmap.pdf", bbox_inches="tight"
        )
        plt.close("all")
