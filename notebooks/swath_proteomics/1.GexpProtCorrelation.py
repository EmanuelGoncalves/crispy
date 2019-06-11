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
from scipy.stats import gaussian_kde
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.GIPlot import GIPlot
from scipy.stats import normaltest, shapiro
from statsmodels.stats.multitest import multipletests
from notebooks.swath_proteomics.SLethal import SLethal


# Paths

logger = logging.getLogger("Crispy")
dpath = pkg_resources.resource_filename("crispy", "data")
rpath = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# Cancer Driver Genes

cgenes = pd.read_csv(f"{dpath}/cancer_genes_latest.csv.gz")


# Protein abundance attenuation

p_attenuated = pd.read_csv(f"{dpath}/protein_attenuation_table.csv", index_col=0)


# Genetic interactions

gis = pd.read_csv(f"{dpath}/sl/gis_gexp_prot.csv.gz", compression="gzip")


# GExp ~ Prot correlation

gp_corr = (
    pd.read_csv(f"{dpath}/sl/prot_gexp_corr.csv.gz")
    .query("dtype == 'no_imputation'")
    .set_index("gene")
)
gp_corr["attenuation"] = (
    p_attenuated.reindex(gp_corr.index)["attenuation_potential"]
    .replace(np.nan, "NA")
    .values
)


# Import data-sets

slethal = SLethal(use_crispr=False, filter_crispr_kws=dict())


# Overlapping genes

genes = set.intersection(set(slethal.gexp.index), set(slethal.prot.index))
logger.info(f"Genes={len(genes)}")


# Protein attenuation

pal = dict(zip(*(["Low", "High", "NA"], CrispyPlot.PAL_DBGD.values())))

plt.figure(figsize=(1, 1.5), dpi=600)
sns.boxplot(
    x="attenuation",
    y="corr",
    notch=True,
    data=gp_corr.query("dtype == 'no_imputation'"),
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    medianprops=CrispyPlot.MEDIANPROPS,
    flierprops=CrispyPlot.FLIERPROPS,
    palette=pal,
    showcaps=False,
    saturation=1,
)
plt.xlabel("Attenuation")
plt.ylabel("Transcript ~ Protein")
plt.axhline(0, ls="-", lw=0.1, c="black", zorder=0)
plt.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)
plt.savefig(f"{rpath}/protein_attenuation_boxplot.pdf", bbox_inches="tight")
plt.close("all")


#

gene = "VIM"

plot_df = pd.concat(
    [
        slethal.prot.loc[gene].rename("protein"),
        slethal.gexp.loc[gene].rename("transcript"),
    ],
    axis=1,
    sort=False,
).dropna()

grid = GIPlot.gi_regression("protein", "transcript", plot_df)
grid.set_axis_labels(f"Protein", f"Transcript")
grid.ax_marg_x.set_title(gene)
plt.savefig(f"{rpath}/protein_gexp_scatter_{gene}.pdf", bbox_inches="tight")
plt.close("all")


#

sl_pairs = [
    ("ARID1A", "ARID1B"),
    ("SMARCA4", "SMARCA2"),
    ("NRAS", "SHOC2"),
    ("ERBB2", "PIK3CA"),
]

# gene_x, gene_y = "CTNNB1", "CTNNB1"
for gene_x, gene_y in sl_pairs:
    plot_df = pd.concat(
        [
            slethal.prot.loc[gene_x].rename("protein"),
            slethal.gexp.loc[gene_x].rename("transcript"),
            slethal.crispr.loc[gene_y].rename("crispr"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_regression("protein", "crispr", plot_df)
    grid.set_axis_labels(f"Protein {gene_x}", f"Essentiality {gene_y}")
    plt.savefig(
        f"{rpath}/protein_essentiality_scatter_{gene_x}_{gene_y}.pdf",
        bbox_inches="tight",
    )
    plt.close("all")

    grid = GIPlot.gi_regression("transcript", "crispr", plot_df)
    grid.set_axis_labels(f"Transcript {gene_x}", f"Essentiality {gene_y}")
    plt.savefig(
        f"{rpath}/transcript_essentiality_scatter_{gene_x}_{gene_y}.pdf",
        bbox_inches="tight",
    )
    plt.close("all")


#

sl_pairs = [
    ("ARID1A", "ARID1B"),
    ("SMARCA4", "SMARCA2"),
    ("NRAS", "SHOC2"),
    ("ERBB2", "PIK3CA"),
    ("BRAF", "SHOC2"),
    ("RB1", "CDK6"),
]

# gene_x, gene_y = "RB1", "SKP2"
for gene_x, gene_y in sl_pairs:
    plot_df = pd.concat(
        [
            slethal.wes.loc[gene_x].rename("protein"),
            slethal.crispr.loc[gene_y].rename("crispr"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = GIPlot.gi_classification(
        "protein", "crispr", plot_df, pal=CrispyPlot.PAL_DBGD
    )
    grid.set_axis_labels(f"{gene_x}\nWES", f"{gene_y}\nEssentiality")
    plt.savefig(
        f"{rpath}/wes_essentiality_scatter_{gene_x}_{gene_y}.pdf", bbox_inches="tight"
    )
    plt.close("all")
