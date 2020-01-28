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
import matplotlib.pyplot as plt
from crispy.GIPlot import GIPlot
from crispy.MOFA import MOFA, MOFAPlot
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
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

cn = cn_obj.filter(subset=samples.intersection(ss.index))
cn = cn.loc[cn.std(1) > 1]
cn = np.log2(cn.divide(ss.loc[cn.columns, "ploidy"]) + 1)
LOG.info(f"Copy-Number: {cn.shape}")

wes = wes_obj.filter(subset=samples, min_events=3, recurrence=True)
wes = wes.loc[wes.std(1) > 0]
LOG.info(f"WES: {wes.shape}")

crispr = crispr_obj.filter(subset=samples, abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")


# MOFA
#

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy),
    iterations=1500,
    convergence_mode="slow",
    factors_n=30,
)

# Export results
mofa.factors.to_csv(f"{RPATH}/1.MultiOmics_factors.csv.gz", compression="gzip")
for m in mofa.weights:
    mofa.weights[m].to_csv(
        f"{RPATH}/1.MultiOmics_weights_{m}.csv.gz", compression="gzip"
    )
mofa.rsquare.to_csv(f"{RPATH}/1.MultiOmics_rsquared.csv.gz", compression="gzip")


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


# Variance explained across data-sets
#

MOFAPlot.variance_explained_heatmap(mofa)
plt.savefig(f"{RPATH}/1.MultiOmics_rsquared_heatmap.pdf", bbox_inches="tight")
plt.close("all")


# Factors data-frame integrated with other measurements
#

covariates = pd.concat(
    [
        prot_obj.sample_corrs.loc[samples].rename("prot_corr"),
        prot.count().loc[samples].rename("prot_meas"),
        pd.get_dummies(ss["growth_properties"]).loc[samples],
        (ss["tissue"] == "Haematopoietic and Lymphoid")
        .astype(int)
        .loc[samples]
        .rename("heme"),
    ],
    axis=1,
).dropna()

df = pd.concat(
    [
        mofa.factors,
        mofa.views["transcriptomics"]
        .loc[["CDH1", "VIM"]]
        .T.add_suffix("_transcriptomics"),
        mofa.views["methylation"].loc[["PTPRT", "GPC5", "SLC5A1"]].T.add_suffix("_methylation"),
        mofa.views["proteomics"].loc[["VIM"]].T.add_suffix("_proteomics"),
        crispr.loc[["HNRNPA1L2", "FH", "ELP4", "ELOVL1"]].T.add_suffix("_crispr"),
        covariates,
        pd.get_dummies(ss.loc[samples, "msi_status"]),
        ss.loc[samples, "msi_status"],
        ss.loc[samples, ["ploidy", "mutational_burden"]],
    ],
    axis=1,
    sort=False,
).dropna()
print(df.corr()["F8"].sort_values())


# Factor 1 associations
#

f = 1
plot_df = pd.concat(
    [
        mofa.factors[f"F{f}"],
        (ss["tissue"] == "Haematopoietic and Lymphoid").astype(int).rename("Haem."),
        ss["growth_properties"],
    ],
    axis=1,
    sort=False,
).dropna()

# Heme
ax = GIPlot.gi_classification(
    f"F{f}", "Haem.", plot_df, orient="h", palette=GIPlot.PAL_DTRACE
)
ax.set_xlabel(f"Factor {f}\n")
ax.set_ylabel(f"Haem cell lines")
plt.gcf().set_size_inches(1.5, 1.0)
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_heme_boxplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

# Growth conditions
ax = GIPlot.gi_classification(
    f"F{f}",
    "growth_properties",
    plot_df,
    orient="h",
    palette=GIPlot.PAL_GROWTH_CONDITIONS,
)
ax.set_xlabel(f"Factor {f}\n")
ax.set_ylabel(f"")
plt.gcf().set_size_inches(1.5, 1.25)
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_growth_conditions_boxplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Factors 2 and 3 associations
#

f_x, f_y = 2, 3
plot_df = pd.concat(
    [
        mofa.factors[[f"F{f_x}", f"F{f_y}"]],
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
        grid = GIPlot.gi_regression(f"F{f}", v, plot_df)
        grid.set_axis_labels(f"Factor {f}", v.replace("_", " "))
        plt.savefig(
            f"{RPATH}/1.MultiOmics_factor{f}_{v}_regression.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

# Tissue plot
ax = GIPlot.gi_tissue_plot(f"F{f_x}", f"F{f_y}", plot_df)
ax.set_xlabel(f"Factor {f_x} (Total R2={mofa.rsquare[f'F{f_x}'].sum() * 100:.1f}%)")
ax.set_ylabel(f"Factor {f_y} (Total R2={mofa.rsquare[f'F{f_y}'].sum() * 100:.1f}%)")
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f_x}_factor{f_y}_tissue_plot.pdf",
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
    ax = GIPlot.gi_continuous_plot(
        f"F{f_x}", f"F{f_y}", z, plot_df, cbar_label=z.replace("_", " ")
    )
    ax.set_xlabel(f"Factor {f_x} (Total R2={mofa.rsquare[f'F{f_x}'].sum() * 100:.1f}%)")
    ax.set_ylabel(f"Factor {f_y} (Total R2={mofa.rsquare[f'F{f_y}'].sum() * 100:.1f}%)")
    plt.savefig(
        f"{RPATH}/1.MultiOmics_F{f_x}_F{f_y}_continous{z}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


# Factors 5 and 10
#

f_x, f_y = 5, 10
plot_df = pd.concat(
    [
        mofa.factors[[f"F{f_x}", f"F{f_y}"]],
        prot_obj.sample_corrs.loc[samples].rename("Protein Reps Corr."),
        prot.count().loc[samples].rename("Protein num. meas."),
    ],
    axis=1,
    sort=False,
).dropna()

for f in [f_x, f_y]:
    for v in ["Protein Reps Corr.", "Protein num. meas."]:
        grid = GIPlot.gi_regression(f"F{f}", v, plot_df)
        grid.set_axis_labels(f"Factor {f}", v.replace("_", " "))
        plt.savefig(
            f"{RPATH}/1.MultiOmics_factor{f}_{v}_regression.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")


# Factor 6
#

f = 6
plot_df = pd.concat(
    [
        mofa.factors[f"F{f}"],
        mofa.views["methylation"].loc[["PTPRT"]].T.add_suffix("_methylation"),
    ],
    axis=1,
    sort=False,
).dropna()

v = "PTPRT_methylation"
grid = GIPlot.gi_regression(f"F{f}", v, plot_df)
grid.set_axis_labels(f"Factor {f}", v.replace("_", " "))
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_{v}_regression.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

# Top features
v = "methylation"

MOFAPlot.factor_weights_scatter(mofa, v, f"F{f}")
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_top_features_{v}.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

MOFAPlot.view_heatmap(
    mofa,
    v,
    f"F{f}",
    col_colors=CrispyPlot.get_palettes(samples, ss),
    title=f"{v} heatmap of Factor{f} top features",
)
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_heatmap_{v}.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")


# Factor 7
#

f = 7
plot_df = pd.concat(
    [
        mofa.factors[f"F{f}"],
        mofa.views["methylation"].loc[["SLC5A1", "GPC5"]].T.add_suffix("_methylation"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

for v in ["SLC5A1_methylation", "GPC5_methylation"]:
    grid = GIPlot.gi_regression(f"F{f}", v, plot_df)
    grid.set_axis_labels(f"Factor {f}", v.replace("_", " "))
    plt.savefig(
        f"{RPATH}/1.MultiOmics_factor{f}_{v}_regression.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

# Top features
v = "methylation"

MOFAPlot.factor_weights_scatter(mofa, v, f"F{f}")
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_top_features_{v}.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

MOFAPlot.view_heatmap(
    mofa,
    v,
    f"F{f}",
    col_colors=CrispyPlot.get_palettes(samples, ss),
    title=f"{v} heatmap of Factor{f} top features",
)
plt.savefig(
    f"{RPATH}/1.MultiOmics_factor{f}_heatmap_{v}.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")
