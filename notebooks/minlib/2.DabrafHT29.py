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
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from minlib import dpath, rpath, LOG
from scipy.interpolate import interpn
from crispy.CRISPRData import CRISPRDataSet
from sklearn.metrics import mean_squared_error


# KY v1.0 & v1.1 sgRNA metrics

lib_metrics = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0)


# HT-29 CRISPR-Cas9 + Dabrafenib timecourse (Day 8, 10, 14, 18 and 21)

dabraf_data = CRISPRDataSet("HT29_Dabraf")
dabraf_count = dabraf_data.counts.remove_low_counts(dabraf_data.plasmids)
dabraf_ss = pd.read_csv(
    f"{dpath}/crispr_manifests/HT29_Dabraf_samplesheet.csv.gz", index_col="sample"
)

# Export gene fold-changes

dabraf_fc_gene = {}
for m in [2, 3, "all"]:
    label = m if m == "all" else f"top{m}"
    LOG.info(f"Exporting gene fold-changes: {m}")

    if m == "all":
        m_fc = dabraf_count[dabraf_count.index.isin(lib_metrics.index)]

    else:
        m_guides = lib_metrics.sort_values("ks_control_min").groupby("Gene").head(m)
        m_fc = dabraf_count[dabraf_count.index.isin(m_guides.index)]

    m_fc = m_fc.norm_rpm().foldchange(dabraf_data.plasmids)

    m_lib = lib_metrics.loc[m_fc.index, "Gene"]
    m_fc = m_fc.loc[m_lib.index].groupby(m_lib).mean()

    dabraf_fc_gene[label] = m_fc.round(5)
    dabraf_fc_gene[label].to_csv(
        f"{rpath}/HT29_Dabraf_gene_fc_{label}.csv.gz", compression="gzip"
    )


# Limma differential essential gene fold-changes

dabraf_limma = pd.concat(
    [
        pd.read_csv(f"{rpath}/HT29_Dabraf_limma_{m}.csv").assign(metric=m)
        for m in ["top2", "top3", "all"]
    ]
)


# Total number of significant differential essentiality

plot_df = (
    dabraf_limma[dabraf_limma["adj.P.Val"] < 0.05]["metric"]
    .value_counts()
    .reset_index()
)

plt.figure(figsize=(2.5, 1), dpi=600)
sns.barplot("metric", "index", data=plot_df, orient="h", color="#959595", linewidth=0)
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
plt.xlabel("Number of signficant dependencies\n(FDR < 5%)")
plt.ylabel("Number of\nsgRNAs")
plt.title("HT-29 CRISPR + Dabrafenib")
plt.savefig(f"{rpath}/HT29_Dabraf_count_signif_assoc.pdf", bbox_inches="tight")
plt.close("all")


# Limma scatter plots

dif_cols = ["Dif10d", "Dif14d", "Dif18d", "Dif21d"]

x_min = dabraf_limma[dif_cols].mean(1).min()
x_max = dabraf_limma[dif_cols].mean(1).max()

y_var = dabraf_limma.query(f"metric == 'all'").set_index(["Gene"])
y = y_var[dif_cols].mean(1)

f, axs = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(3.0, 1.5), dpi=600)

for i, m in enumerate(["top2", "top3"]):
    ax = axs[i]

    x_var_m = dabraf_limma.query(f"metric == '{m}'").set_index(["Gene"])
    x = x_var_m[dif_cols].mean(1).loc[y.index]

    var_fdr = pd.concat(
        [x_var_m["adj.P.Val"].rename("x"), y_var["adj.P.Val"].rename("y")],
        axis=1,
        sort=False,
    )
    var_fdr["signif"] = (var_fdr < 0.05).sum(1)
    var_fdr["size"] = (var_fdr["signif"] + 1) * 3
    var_fdr["color"] = var_fdr["signif"].replace(
        {0: "#e1e1e1", 1: "#ffbb78", 2: "#ff7f0e"}
    )
    var_fdr["marker"] = [
        "o" if s != 1 else ("v" if x < 0.1 else "<")
        for x, y, s in var_fdr[["x", "y", "signif"]].values
    ]

    data, x_e, y_e = np.histogram2d(x, y, bins=20)
    z_var = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    ax.scatter(
        x,
        y,
        c=z_var,
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=var_fdr.loc[x_var_m.index, "size"],
        alpha=0.3,
    )

    for n, df in var_fdr.groupby("signif"):
        if n == 1:
            for marker, df_m in df.groupby("marker"):
                ax.scatter(
                    x.loc[df_m.index],
                    y.loc[df_m.index],
                    c=df_m.iloc[0]["color"],
                    marker=marker,
                    edgecolor="white",
                    lw=0.1,
                    s=1,
                    alpha=1,
                    zorder=2,
                    label="Partial" if n == 1 else "Both",
                )

    rmse = sqrt(mean_squared_error(x, y))
    cor, _ = spearmanr(x, y)
    annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"
    ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

    lims = [x_max, x_min]
    ax.plot(lims, lims, "k-", lw=0.3, zorder=0)
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    ax.set_title(m)
    ax.set_xlabel("Mean fold-changes")
    ax.set_ylabel("All Guides\nMean fold-changes" if i == 0 else None)

    ax.legend(loc=2, frameon=False, prop={"size": 4})

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/HT29_Dabraf_assoc_scatter.png", bbox_inches="tight")
plt.close("all")


# Fold-change examples

genes = ["GRB2", "CSK", "EGFR", "ERBB2", "STK11", "SHC1", "NF1", "PTEN"]

conditions = [("DMSO", "#E1E1E1"), ("Dabraf", "#d62728")]

f, axs = plt.subplots(
    3, len(genes), sharex="all", sharey="all", figsize=(len(genes) * 1.5, 4.5), dpi=600
)

for i, metric in enumerate(["all", "top2", "top3"]):
    for j, gene in enumerate(genes):
        ax = axs[i, j]

        plot_df = pd.concat(
            [dabraf_fc_gene[metric].loc[gene], dabraf_ss], axis=1, sort=False
        )
        plot_df = pd.concat(
            [
                plot_df.query("medium != 'Initial'"),
                plot_df.query("medium == 'Initial'").replace(
                    {"medium": {"Initial": "Dabraf"}}
                ),
                plot_df.query("medium == 'Initial'").replace(
                    {"medium": {"Initial": "DMSO"}}
                ),
            ]
        )

        for mdm, clr in conditions:
            plot_df_mrk = plot_df.query(f"medium == '{mdm}'")
            plot_df_mrk = plot_df_mrk.assign(
                time=plot_df_mrk["time"].apply(lambda x: int(x[1:]))
            )
            plot_df_mrk = plot_df_mrk.groupby("time")[gene].agg([min, max, np.mean])
            plot_df_mrk = plot_df_mrk.assign(time=[f"D{i}" for i in plot_df_mrk.index])

            ax.errorbar(
                plot_df_mrk.index,
                plot_df_mrk["mean"],
                c=clr,
                fmt="--o",
                label=mdm,
                capthick=0.5,
                capsize=2,
                elinewidth=1,
                yerr=[
                    plot_df_mrk["mean"] - plot_df_mrk["min"],
                    plot_df_mrk["max"] - plot_df_mrk["mean"],
                ],
            )

            ax.set_xticks(list(plot_df_mrk.index))

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        ax.set_xlabel("Days" if i == 2 else "")
        ax.set_ylabel(f"sgRNA {metric}\nfold-change" if j == 0 else "")
        ax.set_title(f"{gene}" if i == 0 else "")
        ax.legend(loc=2, frameon=False, prop={"size": 4})

        g_limma = dabraf_limma.query(f"metric == '{metric}'").set_index("Gene").loc[gene]
        annot_text = f"Mean FC={g_limma[dif_cols].mean():.1f}; FDR={g_limma['adj.P.Val']:.1e}"
        ax.text(0.05, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="left")

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/HT29_Dabraf_assoc_examples.pdf", bbox_inches="tight")
plt.close("all")
