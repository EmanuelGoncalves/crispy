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

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import Proteomics, Sample


DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# GDSC baseline cancer cell lines SWATH-Proteomics analysis
#

proteomics = Proteomics()

protein_reps = proteomics.get_data(average_replicates=False)
protein = proteomics.get_data(average_replicates=True)


#
#

plot_df = protein.T.fillna(protein.mean(axis=1)).T

ss = Sample().samplesheet
col_colors = pd.Series({s: CrispyPlot.PAL_TISSUE_2[t] for s, t in ss.loc[plot_df.columns, "tissue"].iteritems()})

g = sns.clustermap(
    plot_df, col_colors=col_colors, cmap="viridis", figsize=(10, 6), mask=protein.isna(), xticklabels=False, yticklabels=False
)

g.ax_heatmap.set_xlabel(None)
g.ax_heatmap.set_ylabel(None)

handles = [
    mpatches.Circle((0.0, 0.0), 0.25, facecolor=v, label=k) for k, v in CrispyPlot.PAL_TISSUE_2.items()
]
g.ax_col_dendrogram.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 0.5), frameon=False)

plt.savefig(f"{RPATH}/0.QC_protein_clustermap.png", bbox_inches="tight", dpi=600)
plt.close("all")


# Sample correlation across cell lines, cancer type and machine
#

samples_corr = protein_reps.corr()
samples_corr.columns.name = "sample2"
samples_corr.index.name = "sample1"

keep = np.triu(np.ones(samples_corr.shape), 1).astype("bool").reshape(samples_corr.size)
samples_corr = samples_corr.stack()[keep].reset_index()

samples_corr = samples_corr.rename(columns={0: "pearson"})

headers = ["Cell_line", "Tissue_type", "Cancer_type", "Instrument", "SIDM"]
samples_corr = pd.concat(
    [
        samples_corr.reset_index(drop=True),
        proteomics.manifest.loc[samples_corr["sample1"], headers]
        .add_suffix("_1")
        .reset_index(drop=True),
        proteomics.manifest.loc[samples_corr["sample2"], headers]
        .add_suffix("_2")
        .reset_index(drop=True),
    ],
    axis=1,
    sort=False,
)

samples_corr["same_cell_line"] = (
    samples_corr["Cell_line_1"] == samples_corr["Cell_line_2"]
).astype(int)
samples_corr["same_cancer_type"] = (
    samples_corr["Tissue_type_1"] == samples_corr["Tissue_type_2"]
).astype(int)
samples_corr["same_machine"] = (
    samples_corr["Instrument_1"] == samples_corr["Instrument_2"]
).astype(int)

samples_corr.to_csv(
    f"{RPATH}/0.QC_protein_corr.csv.gz", compression="gzip", index=False
)
samples_corr.query("same_cell_line == 1").to_csv(f"{DPATH}/proteomics/replicates_corr.csv.gz")


# Sample correlation boxplots
#

comparisons = ["same_cell_line", "same_cancer_type", "same_machine"]

plot_df = pd.melt(samples_corr, id_vars="pearson", value_vars=comparisons).query(
    "value == 1"
)
plot_df = plot_df.append(samples_corr[["pearson"]].assign(variable="all"))

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)
sns.violinplot(
    "pearson",
    "variable",
    data=plot_df,
    cut=0,
    linewidth=0.5,
    palette=list(CrispyPlot.PAL_GROWTH_CONDITIONS.values()),
    ax=ax,
)

y_labels = []
for item in ax.get_yticklabels():
    l = item.get_text()
    r = plot_df.query(f"variable == '{l}'")["pearson"].mean()
    y_labels.append(f"{l.replace('_', ' ')} (mean R={r:.2f})")

ax.set_yticklabels(y_labels)
ax.set_xlabel("Pearson's R")
ax.set_ylabel("")
ax.grid(True, axis="x", ls="-", lw=0.1, alpha=1.0, zorder=0)
plt.savefig(
    f"{RPATH}/0.QC_samples_correlation.pdf", bbox_inches="tight", transparent=True
)
plt.close("all")


# Cumulative sum of number of measurements per protein
#

plot_df = protein.count(1).value_counts().rename("n_proteins")
plot_df = plot_df.reset_index().rename(columns=dict(index="n_measurements"))
plot_df = pd.DataFrame(
    [
        dict(
            perc=np.round(v, 1),
            n_samples=int(np.round(v * protein.shape[1], 0)),
            n_proteins=plot_df.query(f"n_measurements >= {v * protein.shape[1]}")[
                "n_proteins"
            ].sum(),
        )
        for v in np.arange(0, 1.1, 0.1)
    ]
)

_, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=600)

ax.bar(
    plot_df["perc"],
    plot_df["n_proteins"],
    width=.1,
    linewidth=.3,
    color=CrispyPlot.PAL_DBGD[0],
)

for i, row in plot_df.iterrows():
    plt.text(
        row["perc"],
        row["n_proteins"] * 0.99,
        ">0" if row["n_samples"] == 0 else str(int(row["n_samples"])),
        va="top",
        ha="center",
        fontsize=5,
        zorder=10,
        rotation=90,
        color="white",

    )

ax.set_xlabel("Percentage of cell lines")
ax.set_ylabel("Number of proteins")
ax.set_title("Number of quantified proteins")
ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.savefig(
    f"{RPATH}/0.QC_protein_measurements_cumsum.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.close("all")


# Number of cell lines per tissue
#

for t in ["cancer_type", "tissue"]:
    plot_df = Sample().samplesheet.loc[protein.columns, t].value_counts()
    plot_df = plot_df.rename("count").reset_index()

    _, ax = plt.subplots(1, 1, figsize=(2.5, 0.1 * plot_df.shape[0]), dpi=600)

    sns.barplot("count", "index", data=plot_df, orient="h", color=CrispyPlot.PAL_DTRACE[0], saturation=1, linewidth=0, ax=ax)

    ax.set_xlabel("Number of cell lines")
    ax.set_ylabel("")

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
    plt.savefig(
        f"{RPATH}/0.QC_{t}_barplot.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
