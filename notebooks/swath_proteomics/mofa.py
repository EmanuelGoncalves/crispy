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

import gseapy
import logging
import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from crispy.Enrichment import Enrichment
from swath_proteomics.LMModels import LMModels
from data.GIPlot import GIPlot, MOFAPlot
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
    set(crispr.get_data()),
    set(methy.get_data()),
    set(wes.get_data()),
)
LOG.info(f"Samples: {len(samples)}")


# Filter data-sets
#
prot = prot.filter(subset=samples, perc_measures=0.75, replicate_thres=0.7)
LOG.info(f"Proteomics: {prot.shape}")

gexp = gexp.filter(subset=samples)
gexp = gexp.loc[gexp.std(1) > 1]
LOG.info(f"Transcriptomics: {gexp.shape}")

methy = methy.filter(subset=samples)
methy = methy.loc[methy.std(1) > 0.1]
LOG.info(f"Methylation: {methy.shape}")

crispr = crispr.filter(subset=samples, abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")

wes = wes.filter(subset=samples, min_events=5)
LOG.info(f"WES: {wes.shape}")


# Batch effects on proteomics
#
prot_corr = Proteomics().rep_corrs.groupby("SIDM_1")["pearson"].mean()

x_factor, y_factor = "F7", "F9"

factors_exp = pd.concat(
    [
        factors[[x_factor, y_factor]],
        ss[
            [
                "growth",
                "ploidy",
                "mutational_burden",
                "growth_properties",
                "cancer_type",
            ]
        ],
        prot_corr,
        prot.count().rename("measurements"),
    ],
    axis=1,
    sort=False,
).loc[samples]

grid = GIPlot.gi_regression(x_factor, y_factor, factors_exp)
grid.set_axis_labels(
    f"{x_factor} (Total R2={r2[x_factor].sum() * 100:.1f}%)",
    f"{y_factor} (Total R2={r2[y_factor].sum() * 100:.1f}%)",
)
plt.savefig(f"{RPATH}/mofa_scatter_prot_{x_factor}_{y_factor}.pdf", bbox_inches="tight")
plt.close("all")

grid = GIPlot.gi_regression(x_factor, "pearson", factors_exp)
grid.set_axis_labels(
    f"{x_factor} (Total R2={r2[x_factor].sum() * 100:.1f}%)", f"Replicates correlation"
)
plt.savefig(f"{RPATH}/mofa_scatter_prot_{x_factor}_reps.pdf", bbox_inches="tight")
plt.close("all")

grid = GIPlot.gi_regression(y_factor, "measurements", factors_exp)
grid.set_axis_labels(
    f"{y_factor} (Total R2={r2[y_factor].sum() * 100:.1f}%)", f"Number of measurements"
)
plt.savefig(f"{RPATH}/mofa_scatter_prot_{y_factor}_measu.pdf", bbox_inches="tight")
plt.close("all")


#
#

MOFAPlot.variance_explained_heatmap(r2)
plt.savefig(f"{RPATH}/mofa_var_explained_heatmap.pdf", bbox_inches="tight")
plt.close("all")


#
#
for dataset in weights:
    MOFAPlot.factors_weights(weights, dataset)
    plt.savefig(f"{RPATH}/mofa_factor_weights_{dataset}.pdf", bbox_inches="tight")
    plt.close("all")


#
#
factors_lmm_wes = LMModels(
    y=factors, x=wes.T, transform_x=None, institute=False
).matrix_lmm()
print(factors_lmm_wes.query("fdr < 0.05").head(60))

factors_lmm_crispr = LMModels(y=factors, x=crispr.T).matrix_lmm()
print(factors_lmm_crispr.query("fdr < 0.05").head(60))


#
#
n_features = 30
for factor in ["F4"]:
    for i, dataset in enumerate(view_labels):
        # Weights scatter
        MOFAPlot.factor_weights_scatter(
            weights, dataset, factor, r2=r2, n_features=n_features
        )
        plt.savefig(
            f"{RPATH}/mofa_factor_weights_scatter_{dataset}_{factor}.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        # Data heatmaps
        df = datasets[i].loc[
            MOFAPlot.get_top_features(weights, dataset, factor, n_features).index
        ]

        col_colors = pd.DataFrame(
            {
                s: dict(
                    tissue=MOFAPlot.PAL_TISSUE_2[ss.loc[s, "tissue"]],
                    media=MOFAPlot.PAL_GROWTH_CONDITIONS[
                        ss.loc[s, "growth_properties"]
                    ],
                )
                for s in samples
            }
        ).T

        MOFAPlot.data_heatmap(
            df.replace(np.nan, 0.5 if dataset == "methy" else 0),
            mask=df.isnull(),
            col_colors=col_colors,
            title=f"Data-set: {dataset}; Factor: {factor}; Top {n_features} features",
            center=0.5 if dataset == "methy" else 0,
        )
        plt.savefig(
            f"{RPATH}/mofa_data_heatmap_{dataset}_{factor}_top{n_features}.pdf",
            bbox_inches="tight",
        )
        plt.close("all")


#
#
factor, gene, marginal = "F4", "TP63", "STARD3"
plot_df = pd.concat(
    [
        factors[factor],
        crispr.loc[gene],
        wes.loc[marginal],
        ss[
            [
                "growth",
                "ploidy",
                "mutational_burden",
                "cancer_type",
                "growth_properties",
            ]
        ],
    ],
    axis=1,
    sort=False,
).dropna(subset=[factor, gene, marginal])

grid = GIPlot.gi_regression(gene, factor, plot_df)
grid.set_axis_labels(f"{gene}\nCRISPR (scaled FC)", f"Factor {factor[1:]}")
plt.savefig(f"{RPATH}/mofa_scatter_{gene}_{factor}.pdf", bbox_inches="tight")
plt.close("all")

grid = GIPlot.gi_regression_marginal(gene, factor, marginal, plot_df)
grid.set_axis_labels(f"{gene}\nCRISPR (scaled FC)", f"Factor {factor[1:]}")
plt.savefig(f"{RPATH}/mofa_marginal_{gene}_{factor}.pdf", bbox_inches="tight")
plt.close("all")

grid = GIPlot.gi_classification(factor, "growth_properties", plot_df, palette=GIPlot.PAL_GROWTH_CONDITIONS)
plt.savefig(f"{RPATH}/mofa_boxplot_{factor}_growth_properties.pdf", bbox_inches="tight")
plt.close("all")


#
#
x_factor, y_factor, z_factor = "F4", "F1", "KRAS"

plot_df = pd.concat(
    [
        factors[[x_factor, y_factor]],
        crispr.loc[z_factor],
        ss[["growth", "ploidy", "mutational_burden", "cancer_type", "tissue"]],
    ],
    axis=1,
    sort=False,
).dropna(subset=[x_factor, y_factor])

ax = GIPlot.gi_tissue_plot(x_factor, y_factor, plot_df)
ax.set_xlabel(f"{x_factor} (Total R2={r2[x_factor].sum() * 100:.1f}%)")
ax.set_ylabel(f"{y_factor} (Total R2={r2[y_factor].sum() * 100:.1f}%)")
plt.savefig(
    f"{RPATH}/mofa_dimres_factors_{x_factor}_{y_factor}.pdf", bbox_inches="tight"
)
plt.close("all")

ax = GIPlot.gi_continuous_plot(
    x_factor, y_factor, z_factor, plot_df, cbar_label=f"CRISPR {z_factor}"
)
ax.set_xlabel(f"{x_factor} (Total R2={r2[x_factor].sum() * 100:.1f}%)")
ax.set_ylabel(f"{y_factor} (Total R2={r2[y_factor].sum() * 100:.1f}%)")
plt.savefig(
    f"{RPATH}/mofa_dimres_factors_{x_factor}_{y_factor}_{z_factor}.pdf",
    bbox_inches="tight",
)
plt.close("all")


#
#
dataset, factor = "gexp", "F4"
enr_factor = gseapy.ssgsea(
    weights[dataset][factor],
    processes=4,
    gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/c6.all.v7.0.symbols.gmt"),
    no_plot=True,
)
print(enr_factor.res2d.sort_values("sample1"))
