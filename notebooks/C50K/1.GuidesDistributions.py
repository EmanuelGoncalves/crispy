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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from scipy.stats import mannwhitneyu
from crispy.CRISPRData import CRISPRDataSet, ReadCounts
from C50K import rpath, LOG, define_sgrnas_sets, ky_v11_calculate_gene_fc


# Project Score - Kosuke_Yusa v1.1

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_count.counts.remove_low_counts(ky_v11_count.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_count.plasmids)
)
sgrna_sets = define_sgrnas_sets(ky_v11_count.lib, ky_v11_fc, add_controls=True)


# Broad DepMap19Q2 - Avana

avana_data = CRISPRDataSet("Avana")
avana_fc = (
    avana_data.counts.remove_low_counts(avana_data.plasmids)
    .norm_rpm()
    .foldchange(avana_data.plasmids)
)


# Essential and non-essential fold-change distribution

genesets = {
    "essential": Utils.get_essential_genes(return_series=False),
    "nonessential": Utils.get_non_essential_genes(return_series=False),
}

metrics = [
    "ks_control_min",
    "ks_control",
    "jacks_min",
    "doenchroot",
    "doenchroot_min",
    "median_fc",
    "all",
]

metrics_genesets = {}
for n_guides in [2, 3]:
    metrics_genesets[n_guides] = {}

    for m in metrics:
        LOG.info(f"Metric={m}; n_guides={n_guides}")

        if m == "all":
            m_counts = ReadCounts(
                pd.read_csv(
                    f"{rpath}/KosukeYusa_v1.1_sgrna_counts_all.csv.gz", index_col=0
                )
            )
        else:
            m_counts = ReadCounts(
                pd.read_csv(
                    f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{m}_top{n_guides}.csv.gz",
                    index_col=0,
                )
            )
        m_fc = m_counts.norm_rpm().foldchange(ky_v11_count.plasmids)
        m_fc_gene = ky_v11_calculate_gene_fc(m_fc, ky_v11_count.lib.loc[m_counts.index])

        metrics_genesets[n_guides][m] = {
            g: m_fc_gene.reindex(genesets[g]).median(1).dropna() for g in genesets
        }


# Plot sgRNA sets of Yusa_v1.1

plt.figure(figsize=(2.5, 2), dpi=600)
for c in sgrna_sets:
    sns.distplot(
        sgrna_sets[c]["fc"],
        hist=False,
        label=c,
        kde_kws={"cut": 0, "shade": True},
        color=sgrna_sets[c]["color"],
    )
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("sgRNAs fold-change")
plt.legend(frameon=False)
plt.title("Project Score - KY v1.1")
plt.savefig(f"{rpath}/ky_v11_guides_distributions.pdf", bbox_inches="tight")
plt.show()


# Plot sgRNA sets of Avana

avana_sgrna_set = define_sgrnas_sets(
    avana_data.lib, avana_fc, add_controls=True, dataset_name="Avana"
)

plt.figure(figsize=(2.5, 2), dpi=600)
for c in avana_sgrna_set:
    sns.distplot(
        avana_sgrna_set[c]["fc"],
        hist=False,
        label=c,
        kde_kws={"cut": 0, "shade": True},
        color=avana_sgrna_set[c]["color"],
    )
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("sgRNAs fold-change")
plt.legend(frameon=False)
plt.title("DepMap19Q2 - Avana")
plt.savefig(f"{rpath}/avana_guides_distributions.pdf", bbox_inches="tight")
plt.show()


# Gene sets distributions across metrics

f, axs = plt.subplots(
    2,
    len(metrics),
    sharex="all",
    sharey="all",
    figsize=(len(metrics) * 1.5, 3.),
    dpi=600,
)

for j, n_guides in enumerate([2, 3]):
    for i, m in enumerate(metrics):
        ax = axs[j, i]

        for n in genesets:
            sns.distplot(
                metrics_genesets[n_guides][m][n],
                hist=False,
                label=n,
                kde_kws={"cut": 0, "shade": True},
                color=sgrna_sets[n]["color"],
                ax=ax,
            )
            ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
            ax.set_title(f"{m}" if j == 0 else "")
            ax.set_xlabel("sgRNAs fold-change")
            ax.set_ylabel(f"Top {n_guides} sgRNAs" if i == 0 else None)
            ax.legend(frameon=False, prop={"size": 4}, loc=2)

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_metrics_gene_distributions.png", bbox_inches="tight")
plt.close("all")
