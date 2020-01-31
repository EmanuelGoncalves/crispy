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
    CopyNumber,
    CRISPR,
    Sample,
    WES,
)


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


def nan_spearman(x, y):
    m = pd.concat([x.rename("x"), y.rename("y")], axis=1, sort=False).dropna()
    c, p = spearmanr(m["x"], m["y"])
    return c, p


# Data-sets
#

prot_obj = Proteomics()
methy_obj = Methylation()
gexp_obj = GeneExpression()

wes_obj = WES()
cn_obj = CopyNumber()
crispr_obj = CRISPR()


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

crispr = crispr_obj.filter(subset=samples, abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")

cn = cn_obj.filter(subset=samples.intersection(ss.index))
cn = cn.loc[cn.std(1) > 1]
cn = np.log2(cn.divide(ss.loc[cn.columns, "ploidy"]) + 1)
LOG.info(f"Copy-Number: {cn.shape}")

wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
wes = wes.loc[wes.std(1) > 0]
LOG.info(f"WES: {wes.shape}")


# MOFA
#

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy),
    iterations=2000,
    convergence_mode="slow",
    factors_n=20,
    from_file=f"{RPATH}/1.MultiOmics.hdf5",
)
mofa.save_hdf5(f"{RPATH}/1.MultiOmics.hdf5")


# Factor association analysis
#

lmm_crispr = LMModels(y=mofa.factors, x=crispr.T).matrix_lmm()
lmm_crispr.to_csv(
    f"{RPATH}/1.MultiOmics_lmm_crispr.csv.gz", index=False, compression="gzip"
)
print(lmm_crispr.query("fdr < 0.05").head(60))

lmm_cnv = LMModels(y=mofa.factors, x=cn.T, institute=False).matrix_lmm()
lmm_cnv.to_csv(f"{RPATH}/1.MultiOmics_lmm_cnv.csv.gz", index=False, compression="gzip")
print(lmm_cnv.query("fdr < 0.05").head(60))

lmm_wes = LMModels(
    y=mofa.factors, x=wes.T, transform_x="none", institute=False
).matrix_lmm()
lmm_wes.to_csv(f"{RPATH}/1.MultiOmics_lmm_wes.csv.gz", index=False, compression="gzip")
print(lmm_wes.query("fdr < 0.05").head(60))


#
#

factors_tsne, factors_pca = dim_reduction(mofa.factors.T, pca_ncomps=10, input_pca_to_tsne=False)

dimred = dict(tSNE=factors_tsne, pca=factors_pca)

for ctype, df in dimred.items():
    plot_df = pd.concat(
        [df, ss["tissue"]], axis=1, sort=False
    ).dropna()

    ax = plot_dim_reduction(plot_df, ctype=ctype, palette=CrispyPlot.PAL_TISSUE_2)
    ax.set_title(f"Factors {ctype}")
    plt.savefig(
        f"{RPATH}/1.MultiOmics_factors_{ctype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")


# Factor clustermap
#

MOFAPlot.factors_corr_clustermap(mofa)
plt.savefig(f"{RPATH}/1.MultiOmics_factors_corr_clustermap.pdf", bbox_inches="tight")
plt.close("all")


# Variance explained across data-sets
#

MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/1.MultiOmics_rsquared_heatmap.pdf", bbox_inches="tight")
plt.close("all")


# Factors data-frame integrated with other measurements
#

covariates = pd.concat(
    [
        prot_obj.sample_corrs.rename("Proteomics replicates correlation"),
        prot.count().rename("Proteomics n. measurements"),
        methy.mean().rename("Global methylation"),
        pd.get_dummies(ss["msi_status"]),
        pd.get_dummies(ss["growth_properties"]),
        pd.get_dummies(ss["tissue"])["Haematopoietic and Lymphoid"],
        ss.loc[samples, ["ploidy", "mutational_burden"]],
        prot.loc[["VIM"]].T.add_suffix("_proteomics"),
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        methy.loc[["SLC5A1"]].T.add_suffix("_methylation"),
    ],
    axis=1,
).loc[mofa.factors.index]

covariates_corr = pd.DataFrame(
    {
        f: {c: nan_spearman(mofa.factors[f], covariates[c])[0] for c in covariates}
        for f in mofa.factors
    }
)

covariates_df = pd.concat([covariates, mofa.factors], axis=1, sort=False)


# Covairates correlation heatmap
#

MOFAPlot.covariates_heatmap(covariates_corr, mofa)
plt.savefig(
    f"{RPATH}/1.MultiOmics_factors_covariates_clustermap.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Factor 1 associations
#

f = "F1"
plot_df = pd.concat(
    [
        mofa.factors[f],
        (ss["tissue"] == "Haematopoietic and Lymphoid").astype(int).rename("Haem."),
        ss["growth_properties"],
    ],
    axis=1,
    sort=False,
).dropna()

# Heme
ax = GIPlot.gi_classification(
    f, "Haem.", plot_df, orient="h", palette=GIPlot.PAL_DTRACE
)
ax.set_xlabel(f"Factor {f[1:]}\n")
ax.set_ylabel(f"Haem cell lines")
plt.gcf().set_size_inches(1.5, 1.0)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_heme_boxplot.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")

# Growth conditions
ax = GIPlot.gi_classification(
    f, "growth_properties", plot_df, orient="h", palette=GIPlot.PAL_GROWTH_CONDITIONS
)
ax.set_xlabel(f"Factor {f[1:]}\n")
ax.set_ylabel(f"")
plt.gcf().set_size_inches(1.5, 1.25)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_growth_conditions_boxplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Factors 2 and 3 associations
#

f_x, f_y = "F2", "F3"
plot_df = pd.concat(
    [
        mofa.factors[[f_x, f_y]],
        gexp.loc[["CDH1", "VIM"]].T.add_suffix("_transcriptomics"),
        prot.loc[["VIM"]].T.add_suffix("_proteomics"),
        mofa.views["methylation"].loc[["PTPRT"]].T.add_suffix("_methylation"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

# Regression plots
for f in [f_x, f_y]:
    for v in ["CDH1_transcriptomics", "VIM_transcriptomics", "VIM_proteomics"]:
        grid = GIPlot.gi_regression(f, v, plot_df)
        grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_{v}_regression.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

# Tissue plot
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df)
ax.set_xlabel(f"Factor {f_x[1:]} (Total R2={mofa.rsquare[f_x].sum() * 100:.1f}%)")
ax.set_ylabel(f"Factor {f_y[1:]} (Total R2={mofa.rsquare[f_y].sum() * 100:.1f}%)")
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f_x}_{f_y}_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

# Continous annotation
for z in [
    "VIM_proteomics",
    "VIM_transcriptomics",
    "CDH1_transcriptomics",
    "PTPRT_methylation",
]:
    ax = GIPlot.gi_continuous_plot(f_x, f_y, z, plot_df, cbar_label=z.replace("_", " "))
    ax.set_xlabel(f"Factor {f_x[1:]} (Total R2={mofa.rsquare[f_x].sum() * 100:.1f}%)")
    ax.set_ylabel(f"Factor {f_y[1:]} (Total R2={mofa.rsquare[f_y].sum() * 100:.1f}%)")
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f_x}_{f_y}_continous{z}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Factors 4 and 6
#

f_x, f_y = "F4", "F6"
plot_df = pd.concat(
    [mofa.factors[[f_x, f_y]], covariates["Global methylation"], ss["tissue"]],
    axis=1,
    sort=False,
).dropna()

for f in [f_x, f_y]:
    for v in ["Global methylation"]:
        grid = GIPlot.gi_regression(f, v, plot_df)
        grid.set_axis_labels(f"Factor {f[1:]}", v)
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_{v}_regression.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")


# Factors 5 and 10
#

f_x, f_y = "F5", "F10"
plot_df = pd.concat(
    [
        mofa.factors[[f_x, f_y]],
        prot_obj.sample_corrs.loc[samples].rename("Protein Reps Corr."),
        prot.count().loc[samples].rename("Protein num. meas."),
    ],
    axis=1,
    sort=False,
).dropna()

for f in [f_x, f_y]:
    for v in ["Protein Reps Corr.", "Protein num. meas."]:
        grid = GIPlot.gi_regression(f, v, plot_df)
        grid.set_axis_labels(f"Factor {f[1:]}", v)
        plt.savefig(
            f"{RPATH}/1.MultiOmics_{f}_{v}_regression.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")


# Factor 7
#

f = "F7"
plot_df = pd.concat(
    [
        mofa.factors[f],
        mofa.views["methylation"]
        .loc[["SLC5A1", "POLR2K"]]
        .T.add_suffix("_methylation"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

for v in ["SLC5A1_methylation", "POLR2K_methylation"]:
    grid = GIPlot.gi_regression(f, v, plot_df)
    grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
    plt.savefig(
        f"{RPATH}/1.MultiOmics_{f}_{v}_regression.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

# Top features
v = "methylation"

MOFAPlot.factor_weights_scatter(mofa, v, f)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_top_features_{v}.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

MOFAPlot.view_heatmap(
    mofa,
    v,
    f,
    col_colors=CrispyPlot.get_palettes(samples, ss),
    title=f"{v} heatmap of Factor{f[1:]} top features",
)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_heatmap_{v}.pdf", transparent=True, bbox_inches="tight"
)
plt.close("all")


# Factor 12
#

f = "F12"
plot_df = pd.concat([mofa.factors[f], ss["msi_status"]], axis=1, sort=False).dropna()

ax = GIPlot.gi_classification(
    f, "msi_status", plot_df, orient="h", palette=GIPlot.PAL_MSS
)
ax.set_xlabel(f"Factor {f[1:]}\n")
ax.set_ylabel(f"MSI status")
plt.gcf().set_size_inches(1.5, 1.0)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_msi_status_boxplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Factor 11
#

f = "F11"
plot_df = pd.concat(
    [
        mofa.factors[f],
        prot.loc[["KRT17"]].T.add_suffix("_proteomics"),
        gexp.loc[["TYR"]].T.add_suffix("_transcriptomics"),
        pd.get_dummies(ss["tissue"])["Skin"].apply(lambda v: "Yes" if v else "No"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

ax = GIPlot.gi_classification("Skin", f, plot_df, orient="v", palette=GIPlot.PAL_YES_NO)
ax.set_ylabel(f"Factor {f[1:]}\n")
ax.set_xlabel(f"Skin")
plt.gcf().set_size_inches(0.7, 1.5)
plt.savefig(
    f"{RPATH}/1.MultiOmics_{f}_Skin_boxplot.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")

ax = GIPlot.gi_tissue_plot("TYR_transcriptomics", "KRT17_proteomics", plot_df)
plt.savefig(
    f"{RPATH}/1.MultiOmics_TYR_KRT17_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")
