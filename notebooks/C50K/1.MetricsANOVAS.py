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
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.DataImporter import Mobem
from sklearn.metrics import mean_squared_error
from crispy.CRISPRData import CRISPRDataSet, ReadCounts
from C50K import rpath, ky_v11_calculate_gene_fc, define_sgrnas_sets


# Biomarker associations (ANOVA) using Strongly Selective Dependencies (SSDs)

lm_assocs = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_biomarker.xlsx")


# Genomic data

mobem = Mobem().get_data()


# Project Score samples acquired with Kosuke_Yusa v1.1 library

ky_v11_data = CRISPRDataSet("Yusa_v1.1")
ky_v11_fc = (
    ky_v11_data.counts.remove_low_counts(ky_v11_data.plasmids)
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)

sgrna_sets = define_sgrnas_sets(ky_v11_data.lib, ky_v11_fc, add_controls=True)


# All guides used in metrics counts

metrics_fc_all = ReadCounts(
    pd.read_csv(
        f"{rpath}/KosukeYusa_v1.1_sgrna_counts_all.csv.gz",
        index_col=0,
    )
)
metrics_fc_all = metrics_fc_all.norm_rpm().foldchange(ky_v11_data.plasmids)
metrics_fc_all = ky_v11_calculate_gene_fc(metrics_fc_all, ky_v11_data.lib.loc[metrics_fc_all.index])


# ANOVA scatter plots

metrics = ["ks_control_min", "ks_control", "median_fc", "jacks_min", "doenchroot", "doenchroot_min"]

f, axs = plt.subplots(
    2,
    len(metrics),
    sharex="all",
    sharey="all",
    figsize=(len(metrics) * 1.5, 3.),
    dpi=600,
)

x_min, x_max = lm_assocs["beta"].min(), lm_assocs["beta"].max()

for j, n_guides in enumerate([2, 3]):
    y_var = lm_assocs.query(f"(metric == 'all') & (n_guides == '{n_guides}')").set_index(["gene", "phenotype"])

    for i, m in enumerate(metrics):
        ax = axs[j, i]

        x_var_m = lm_assocs.query(f"(metric == '{m}') & (n_guides == '{n_guides}')").set_index(["gene", "phenotype"])

        var_fdr = pd.concat([x_var_m["fdr"].rename("x"), y_var["fdr"].rename("y")], axis=1)
        var_fdr["signif"] = (var_fdr < 0.1).sum(1)
        var_fdr["size"] = (var_fdr["signif"] + 1) * 3
        var_fdr["color"] = var_fdr["signif"].replace(
            {0: "#e1e1e1", 1: "#ffbb78", 2: "#ff7f0e"}
        )
        var_fdr["marker"] = [
            "o" if s != 1 else ("v" if x < 0.1 else "<")
            for x, y, s in var_fdr[["x", "y", "signif"]].values
        ]

        data, x_e, y_e = np.histogram2d(
            x_var_m["beta"], y_var.loc[x_var_m.index, "beta"], bins=20
        )
        z_var = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([x_var_m["beta"], y_var.loc[x_var_m.index, "beta"]]).T,
            method="splinef2d",
            bounds_error=False,
        )

        ax.scatter(
            x_var_m["beta"],
            y_var.loc[x_var_m.index, "beta"],
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
                        x_var_m.loc[df_m.index, "beta"],
                        y_var.loc[df_m.index, "beta"],
                        c=df_m.iloc[0]["color"],
                        marker=marker,
                        edgecolor="white",
                        lw=0.3,
                        s=df_m.iloc[0]["size"],
                        alpha=1,
                        zorder=2,
                        label="Partial" if n == 1 else "Both",
                    )

        rmse = sqrt(mean_squared_error(x_var_m["beta"], y_var.loc[x_var_m.index, "beta"]))
        cor, _ = spearmanr(x_var_m["beta"], y_var.loc[x_var_m.index, "beta"])
        annot_text = f"Spearman's R={cor:.2g}; RMSE={rmse:.2f}"

        ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

        lims = [x_max, x_min]
        ax.plot(lims, lims, "k-", lw=0.3, zorder=0)
        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

        ax.set_title(m if j == 0 else "")
        ax.set_xlabel("Effect size (beta)")
        ax.set_ylabel("All Guides\nEffect size (beta)" if i == 0 else None)

        ax_inv = ax.twinx()
        ax_inv.set_ylabel(f"Top {n_guides} sgRNAs" if i == 5 else None)
        ax_inv.get_yaxis().set_ticks([])

        if i == 5 and j == 1:
            ax.legend(loc=2, frameon=False, prop={"size": 4})

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{rpath}/ky_v11_anova.png", bbox_inches="tight")
plt.show()


# ANOVA top missed/found biormarker associations

samples = set(mobem).intersection(metrics_fc_all)

for n_guides in [2, 3]:
    # Metric fold-change
    metrics_fc = ReadCounts(
        pd.read_csv(
            f"{rpath}/KosukeYusa_v1.1_sgrna_counts_ks_control_min_top{n_guides}.csv.gz",
            index_col=0,
        )
    )
    metrics_fc = metrics_fc.norm_rpm().foldchange(ky_v11_data.plasmids)
    metrics_fc = ky_v11_calculate_gene_fc(metrics_fc, ky_v11_data.lib.loc[metrics_fc.index])

    # Build dataframe with top missed and found biomarker associations
    n_top = 10

    plot_df = pd.concat(
        [
            lm_assocs.query(f"(metric == '{m}') & (n_guides == '{n_guides}')")
            .set_index(["gene", "phenotype"])
            .add_prefix(f"{m}_")
            for m in ["all", "ks_control_min"]
        ],
        axis=1,
        sort=False,
    )
    plot_df = plot_df[(plot_df[["all_fdr", "ks_control_min_fdr"]] < 0.1).sum(1) == 1]
    plot_df["delta_beta_abs"] = plot_df.eval("all_beta - ks_control_min_beta").abs()
    plot_df = plot_df.sort_values("delta_beta_abs", ascending=False)

    plot_df = [
        (
            "Top missed associations",
            "top_missed",
            plot_df.query("all_fdr < 0.1")
            .head(n_top),
        ),
        (
            "Top found associations",
            "top_found",
            plot_df.query("ks_control_min_fdr < 0.1")
            .head(n_top),
        ),
    ]

    for (assoc_name, assoc_file, assoc_top) in plot_df:
        n_cols = 5
        n_rows = int(np.ceil(n_top / n_cols))

        f, axs = plt.subplots(
            n_rows,
            n_cols,
            sharex="all",
            sharey="all",
            figsize=(n_cols * 1.5, n_rows * 1.5),
            dpi=600,
        )

        for i, ((feature, phenotype), row) in enumerate(assoc_top.iterrows()):
            df = pd.concat(
                [
                    mobem.loc[feature, samples].rename("feature"),
                    metrics_fc.loc[phenotype, samples].rename("ks"),
                    metrics_fc_all.loc[phenotype, samples].rename("all"),
                ],
                axis=1,
                sort=False,
            )
            df = pd.melt(df, id_vars="feature", value_vars=["ks", "all"])

            row_i = int(np.floor(i / n_cols))
            col_i = i % n_cols
            ax = axs[row_i][col_i]

            sns.boxplot(
                "variable",
                "value",
                "feature",
                data=df,
                order=["all", "ks"],
                palette={1: "#d62728", 0: "#E1E1E1"},
                saturation=1,
                showcaps=False,
                showfliers=False,
                ax=ax,
            )
            sns.stripplot(
                "variable",
                "value",
                "feature",
                data=df,
                order=["all", "ks"],
                dodge=True,
                edgecolor="white",
                linewidth=0.1,
                size=2,
                palette={1: "#d62728", 0: "#E1E1E1"},
                ax=ax,
            )

            for g in sgrna_sets:
                ax.axhline(
                    sgrna_sets[g]["fc"].median(),
                    ls="-",
                    lw=0.3,
                    c=sgrna_sets[g]["color"],
                    zorder=0,
                )

            ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)
            ax.set_xlabel("Guide selection" if row_i == (n_rows - 1) else None)
            ax.set_ylabel(f"Fold-change" if col_i == 0 else None)
            ax.set_title(f"{feature} ~ {phenotype}")
            ax.get_legend().remove()

            all_annot_text = f"beta={row['all_beta']:.1f}\nFDR={row['all_fdr']:.2g}"
            ax.text(
                0.3, 0.1, all_annot_text, fontsize=4, transform=ax.transAxes, ha="center"
            )

            ks_annot_text = f"beta={row['ks_control_min_beta']:.1f}\nFDR={row['ks_control_min_fdr']:.2g}"
            ax.text(
                0.8, 0.1, ks_annot_text, fontsize=4, transform=ax.transAxes, ha="center"
            )

        plt.suptitle(assoc_name, y=1.05)
        plt.subplots_adjust(hspace=0.25, wspace=0.05)
        plt.savefig(f"{rpath}/ky_v11_anova_examples_{assoc_file}_top{n_guides}.png", bbox_inches="tight")
        plt.close("all")
