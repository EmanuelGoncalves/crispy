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
from scipy.stats import spearmanr
from crispy.MOFA import MOFA, MOFAPlot
from crispy.Enrichment import Enrichment
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


# Covariates
#

covariates = (
    pd.concat(
        [
            prot_obj.sample_corrs.rename("prot_corr"),
            prot_obj.get_data().count().rename("prot_meas"),
            methy_obj.get_data().mean().rename("global_methylation"),
            pd.get_dummies(ss["growth_properties"]),
            pd.get_dummies(ss["tissue"])["Haematopoietic and Lymphoid"],
        ],
        axis=1,
    )
    .loc[samples]
    .dropna()
)
LOG.info(f"Covariates: {covariates.shape}")


# Filter data-sets
#

prot = prot_obj.filter(subset=set(covariates.index), perc_measures=0.75)
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

crispr = crispr_obj.filter(
    subset=samples, abs_thres=0.5, min_events=3, drop_core_essential=True
)
LOG.info(f"CRISPR: {crispr.shape}")


# MOFA
#

mofa = MOFA(
    views=dict(proteomics=prot, transcriptomics=gexp, methylation=methy),
    iterations=2500,
    covariates=covariates,
    convergence_mode="slow",
    factors_n=100,
)

mofa.factors.to_csv(f"{RPATH}/2.MultiOmicsCovs_factors.csv.gz", compression="gzip")
for m in mofa.weights:
    mofa.weights[m].to_csv(
        f"{RPATH}/2.MultiOmicsCovs_weights_{m}.csv.gz", compression="gzip"
    )
mofa.rsquare.to_csv(f"{RPATH}/2.MultiOmicsCovs_rsquared.csv.gz", compression="gzip")
mofa.save_hdf5(f"{RPATH}/2.MultiOmicsCovs.hdf5")


# Factor association analysis
#

lmm_crispr = LMModels(y=mofa.factors, x=crispr.T).matrix_lmm()
lmm_crispr.to_csv(
    f"{RPATH}/2.MultiOmicsCovs_lmm_crispr.csv.gz", index=False, compression="gzip"
)
print(lmm_crispr.query("fdr < 0.05").head(60))

lmm_cnv = LMModels(y=mofa.factors, x=cn.T, institute=False).matrix_lmm()
lmm_cnv.to_csv(
    f"{RPATH}/2.MultiOmicsCovs_lmm_cnv.csv.gz", index=False, compression="gzip"
)
print(lmm_cnv.query("fdr < 0.05").head(60))

lmm_wes = LMModels(
    y=mofa.factors, x=wes.T, transform_x="none", institute=False
).matrix_lmm()
lmm_wes.to_csv(
    f"{RPATH}/2.MultiOmicsCovs_lmm_wes.csv.gz", index=False, compression="gzip"
)
print(lmm_wes.query("fdr < 0.05").head(60))


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
    f"{RPATH}/2.MultiOmicsCovs_factors_covariates_clustermap.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


#
#

f, c = "F45", "WRN"
plot_df = pd.concat(
    [
        mofa.factors[[f]],
        crispr.loc[[c]].T.add_suffix("_crispr"),
        ss["msi_status"],
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

for v in [f"{c}_crispr"]:
    grid = GIPlot.gi_regression(f, v, plot_df)
    grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f}_{v}_regression.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

ax = GIPlot.gi_classification("msi_status", f, plot_df, palette=GIPlot.PAL_MSS)
ax.set_ylabel(f"Factor {f[1:]}\n")
ax.set_xlabel(f"MSI status")
plt.gcf().set_size_inches(0.7, 1.5)
plt.savefig(
    f"{RPATH}/2.MultiOmicsCovs_{f}_msi_status_boxplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")

for v in mofa.views_labels:
    MOFAPlot.factor_weights_scatter(mofa, v, f)
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f}_top_features_{v}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")


#
#

f, c = "F6", "TP63"
plot_df = pd.concat(
    [
        mofa.factors[[f, "F11"]],
        crispr.loc[[c, "CTNNB1"]].T.add_suffix("_crispr"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

for v in [f"{c}_crispr"]:
    grid = GIPlot.gi_regression(f, v, plot_df)
    grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f}_{v}_regression.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

for v in mofa.views_labels:
    MOFAPlot.factor_weights_scatter(mofa, v, f)
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f}_top_features_{v}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

f_x, f_y = f, "F11"
ax = GIPlot.gi_tissue_plot(f_x, f_y, plot_df)
ax.set_xlabel(f"Factor {f_x[1:]} (Total R2={mofa.rsquare[f_x].sum() * 100:.1f}%)")
ax.set_ylabel(f"Factor {f_y[1:]} (Total R2={mofa.rsquare[f_y].sum() * 100:.1f}%)")
plt.savefig(
    f"{RPATH}/2.MultiOmicsCovs_{f_x}_{f_y}_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")

f_x, f_y = f, "F11"
for v in [f"{c}_crispr", "CTNNB1_crispr"]:
    ax = GIPlot.gi_continuous_plot(
        f_x, f_y, v, plot_df, cbar_label=v.replace("_", " ")
    )
    ax.set_xlabel(f"Factor {f_x[1:]} (Total R2={mofa.rsquare[f_x].sum() * 100:.1f}%)")
    ax.set_ylabel(f"Factor {f_y[1:]} (Total R2={mofa.rsquare[f_y].sum() * 100:.1f}%)")
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f_x}_{f_y}_continous{v}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")

path_enr = mofa.pathway_enrichment(f, permutation_num=1000)

v, gmt, s = "proteomics", "c2.cp.kegg.v7.0.symbols.gmt", "KEGG_MISMATCH_REPAIR"
ax = Enrichment([gmt]).plot(mofa.weights[v][f], gmt, s, vertical_lines=True, shade=True)
ax[1].set_ylabel(v)
plt.gcf().set_size_inches(2.5, 2.5, forward=True)
plt.savefig(
    f"{RPATH}/2.MultiOmicsCovs_{f}_{s}_tissue_plot.pdf",
    transparent=True,
    bbox_inches="tight",
)
plt.close("all")


#
#

f, c = "F39", "TGIF1"
plot_df = pd.concat(
    [
        mofa.factors[[f]],
        crispr.loc[[c]].T.add_suffix("_crispr"),
        ss["tissue"],
    ],
    axis=1,
    sort=False,
).dropna()

for v in [f"{c}_crispr"]:
    grid = GIPlot.gi_regression(f, v, plot_df)
    grid.set_axis_labels(f"Factor {f[1:]}", v.replace("_", " "))
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f}_{v}_regression.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

for v in mofa.views_labels:
    MOFAPlot.factor_weights_scatter(mofa, v, f)
    plt.savefig(
        f"{RPATH}/2.MultiOmicsCovs_{f}_top_features_{v}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close("all")
