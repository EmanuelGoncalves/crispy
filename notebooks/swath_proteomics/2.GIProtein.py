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
from crispy.QCPlot import QCplot
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.GIPlot import GIPlot
from swath_proteomics.LMModels import LMModels
from sklearn.metrics import roc_auc_score, roc_curve
from crispy.DataImporter import Proteomics, CORUM, GeneExpression, CRISPR, Sample


FDR_THRES = 0.05
LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Data-sets
#
ss = Sample().samplesheet

prot_obj = Proteomics()
prot = prot_obj.filter(perc_measures=None)
LOG.info(f"Proteomics: {prot.shape}")

gexp_obj = GeneExpression()
gexp = gexp_obj.filter()
LOG.info(f"Transcriptomics: {gexp.shape}")

crispr_obj = CRISPR()
crispr = crispr_obj.filter()
LOG.info(f"CRISPR: {crispr.shape}")

samples = set.intersection(set(prot), set(gexp), set(crispr))
LOG.info(f"Samples: {len(samples)}")


# Import Protein / Transcriptomics ~ CRISPR associations
#
lmm_gexp = pd.read_csv(f"{RPATH}/lmm_gexp_crispr.csv.gz")
lmm_prot = pd.read_csv(f"{RPATH}/lmm_protein_crispr.csv.gz")


# Assemble protein association data-frame
#

lmm_prot_signif = lmm_prot.query(f"fdr < {FDR_THRES}")

# CORUM interactions
corum = set(CORUM().db_melt_symbol)
lmm_prot_signif = lmm_prot_signif.assign(
    corum=[
        int((p1, p2) in corum) for p1, p2 in lmm_prot_signif[["y_id", "x_id"]].values
    ]
)

# GExp associations
gexp_signif = {
    (y, x) for y, x in lmm_gexp.query(f"fdr < {FDR_THRES}")[["y_id", "x_id"]].values
}
lmm_prot_signif = lmm_prot_signif.assign(
    gexp_signif=[
        int((p1, p2) in gexp_signif)
        for p1, p2 in lmm_prot_signif[["y_id", "x_id"]].values
    ]
)

# Cancer Driver Genes
cgenes = set(pd.read_csv(f"{DPATH}/cancer_genes_latest.csv.gz")["gene_symbol"])
lmm_prot_signif["y_cgene"] = lmm_prot_signif["y_id"].isin(cgenes).astype(int).values
lmm_prot_signif["x_cgene"] = lmm_prot_signif["x_id"].isin(cgenes).astype(int).values


# Representative examples
#

pairs = [("MBD1", "TFAP2C"), ("SMC1A", "STAG1"), ("ERBB2", "ERBB2"), ("NDUFV1", "GPI")]

prot_y, crispr_x = "NDUFS7", "GPI"

for prot_y, crispr_x in pairs:
    plot_df = (
        pd.concat(
            [
                crispr.loc[crispr_x, samples].rename("crispr"),
                prot.loc[prot_y, samples].rename("protein"),
                ss[["model_name", "cancer_type"]],
            ],
            axis=1,
            sort=False,
        )
        .dropna()
        .sort_values("crispr")
    )

    grid = GIPlot.gi_regression("crispr", "protein", plot_df)
    grid.set_axis_labels(
        f"{crispr_x}\nCRISPR (scaled FC)", f"{prot_y}\nProtein (mean intensity)"
    )
    grid.ax_marg_x.set_title(f"Prot: {prot_y} ~ CRISPR: {crispr_x}")
    plt.savefig(
        f"{RPATH}/lmm_protein_scatter_{crispr_x}_{prot_y}.pdf", bbox_inches="tight"
    )
    plt.close("all")


#
#

crispr_x = "TP63"

df = lmm_prot_signif.query(f"(x_id == '{crispr_x}') & (fdr < .01)")

plot_df = (
    pd.concat(
        [
            crispr.loc[crispr_x, samples].rename("crispr"),
            prot.loc[df["y_id"]].min().rename("protein"),
            ss[["model_name", "cancer_type"]],
        ],
        axis=1,
        sort=False,
    )
    .dropna()
    .sort_values("crispr")
)

grid = GIPlot.gi_regression("crispr", "protein", plot_df)
grid.set_axis_labels(
    f"{crispr_x}\nCRISPR (scaled FC)", f"min\nProtein (mean intensity)"
)
plt.savefig(
    f"{RPATH}/lmm_protein_scatter_{crispr_x}_min.pdf", bbox_inches="tight"
)
plt.close("all")
