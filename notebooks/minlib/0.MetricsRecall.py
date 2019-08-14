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
from crispy.QCPlot import QCplot
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib import rpath, LOG, project_score_sample_map
from sklearn.metrics import recall_score, precision_score


def guides_recall_benchmark(
    metrics, sgrna_counts, dataset, smap, nguides_thres=5, jacks_thres=1, fpr_thres=0.01
):
    # Define set of guides
    LOG.info(f"#(sgRNAs)={metrics.shape[0]}")

    # AROC scores
    scores = []

    for m in ["ks_control_min", "jacks_min", "combined"]:
        LOG.info(f"Metric = {m}")

        for n in range(1, (nguides_thres + 1)):
            # Metric top guides
            if m == "combined":
                metric_topn = (
                    metrics.query(f"jacks_min < {jacks_thres}")
                    .sort_values("ks_control_min")
                    .groupby("Gene")
                    .head(n=n)
                )

            else:
                metric_topn = metrics.sort_values(m).groupby("Gene").head(n=n)

            # Calculate fold-changes on subset
            metric_fc = sgrna_counts.loc[metric_topn.index]
            metric_fc = metric_fc.norm_rpm().foldchange(dataset.plasmids)
            metric_fc = metric_fc.groupby(metrics["Gene"]).mean()
            metric_fc = metric_fc.groupby(smap["model_id"], axis=1).mean()

            # Binarise fold-changes (1% FDR)
            metric_thres = [
                QCplot.aroc_threshold(metric_fc[s], fpr_thres=fpr_thres)[1]
                for s in metric_fc
            ]
            metric_bin = (metric_fc < metric_thres).astype(int)

            genes, samples = set(metric_bin.index), set(metric_bin)
            LOG.info(f"Genes:{len(genes)}; Samples:{len(samples)}")

            # Evaluation
            metric_recalls = pd.DataFrame(
                [
                    dict(
                        sample=s,
                        metric=m,
                        nguides=n,
                        recall=recall_score(
                            ky_bin.loc[genes, s], metric_bin.loc[genes, s]
                        ),
                        precision=precision_score(
                            ky_bin.loc[genes, s], metric_bin.loc[genes, s]
                        ),
                        ess_recall_thres=QCplot.aroc_threshold(
                            metric_fc.loc[genes, s], fpr_thres=fpr_thres
                        )[0],
                        ess_recall=QCplot.recall_curve(metric_fc.loc[genes, s])[2],
                        average_precision=QCplot.precision_recall_curve(metric_fc.loc[genes, s])[0],
                        recall_fdr=QCplot.precision_recall_curve(metric_fc.loc[genes, s])[1]
                    )
                    for s in metric_bin
                ]
            )

            # Store
            scores.append(metric_recalls)

    scores = pd.concat(scores)

    return scores


# Import sgRNA counts and efficiency metrics

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# Selected metrics + Gene annotation + drop NaNs

mlib = (
    Library.load_library("master.csv.gz", set_index=False)
    .query("lib == 'KosukeYusa'")
    .dropna(subset=["Gene"])
    .set_index("sgRNA_ID")
)

mlib = mlib[mlib.index.isin(ky_counts.index)][
    ["ks_control_min", "jacks_min", "Gene"]
].dropna()


# Fold-changes

ky_sgrna_fc = ky_counts.norm_rpm().foldchange(ky.plasmids)

ky_fc = (
    ky_sgrna_fc.groupby(mlib["Gene"]).mean().groupby(ky_smap["model_id"], axis=1).mean()
)

ky_thres = [QCplot.aroc_threshold(ky_fc[s], fpr_thres=0.01)[1] for s in ky_fc]
ky_bin = (ky_fc < ky_thres).astype(int)


# Benchmark sgRNA: Essential/Non-essential AROC

metrics_recall = guides_recall_benchmark(mlib, ky_counts, ky, ky_smap)
metrics_recall.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_recall.xlsx", index=False)

metrics_recall_all = pd.DataFrame(
    {
        s: dict(
            ess_recall=QCplot.recall_curve(ky_fc[s])[2],
            ess_recall_thres=QCplot.aroc_threshold(ky_fc[s], fpr_thres=0.01)[0],
            average_precision=QCplot.precision_recall_curve(ky_fc[s])[0],
        )
        for s in ky_fc
    }
).T


#

plot_df = pd.melt(
    metrics_recall,
    id_vars=["metric", "nguides"],
    value_vars=["ess_recall", "ess_recall_thres", "precision", "recall", "average_precision", "recall_fdr"],
)

n_cols = len(set(plot_df["nguides"]))
n_rows = len(set(plot_df["variable"]))

xorder = ["ks_control_min", "jacks_min", "combined"]
pal = {
    "ks_control_min": CrispyPlot.PAL_DBGD[0],
    "jacks_min": CrispyPlot.PAL_DBGD[2],
    "combined": CrispyPlot.PAL_DBGD[1],
}

f, axs = plt.subplots(
    n_rows, n_cols, sharey="row", sharex="col", dpi=600, figsize=(0.5 * n_cols, n_rows)
)

for i, (var, var_df) in enumerate(plot_df.groupby("variable")):
    for j, (nguides, guide_df) in enumerate(var_df.groupby("nguides")):
        ax = axs[i][j]

        sns.boxplot(
            "metric",
            "value",
            data=guide_df,
            order=xorder,
            palette=pal,
            saturation=1,
            showcaps=False,
            boxprops=dict(linewidth=0.3),
            whiskerprops=dict(linewidth=0.3),
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markersize=1.0,
                linestyle="none",
                markeredgecolor="none",
                alpha=0.6,
            ),
            notch=True,
            ax=ax,
        )

        if var in ["ess_recall", "ess_recall_thres", "average_precision"]:
            y_value = metrics_recall_all[var].median()
            ax.axhline(y_value, ls="-", lw=0.5, c=CrispyPlot.PAL_DBGD[3], zorder=0)

        ax.set_ylabel(var if j == 0 else None)
        ax.set_xlabel(None)
        ax.set_title(f"#(sgRNA)={nguides}" if i == 0 else None, fontsize=4)

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig(f"{rpath}/ky_v11_ks_guides_benchmark_boxplots.png", bbox_inches="tight")
plt.close("all")


# Scatterplot

x_feature = "combined"
y_features = ["jacks_min", "ks_control_min"]

for dtype in ["ess_recall", "ess_recall_thres", "precision", "recall", "average_precision", "recall_fdr"]:
    f, axs = plt.subplots(
        len(y_features),
        metrics_recall["nguides"].max(),
        sharex="all",
        sharey="all",
        figsize=(metrics_recall["nguides"].max(), len(y_features)),
        dpi=600,
    )

    x_min, x_max = metrics_recall[dtype].min(), metrics_recall[dtype].max()

    for i in range(metrics_recall["nguides"].max()):
        plot_df = metrics_recall.query(f"nguides == {(i + 1)}")

        x_var = (
            plot_df.query(f"metric == '{x_feature}'")
            .set_index("sample")[dtype]
            .rename("x")
        )
        y_vars = pd.concat(
            [
                plot_df.query(f"metric == '{f}'")
                .set_index("sample")[dtype]
                .rename(f"y{i + 1}")
                for i, f in enumerate(y_features)
            ],
            axis=1,
        )
        plot_df = pd.concat([x_var, y_vars], axis=1)

        for j in range(len(y_features)):
            data, x_e, y_e = np.histogram2d(plot_df["x"], plot_df[f"y{j + 1}"], bins=20)
            z_var = interpn(
                (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([plot_df["x"], plot_df[f"y{j + 1}"]]).T,
                method="splinef2d",
                bounds_error=False,
            )

            axs[j][i].scatter(
                plot_df["x"],
                plot_df[f"y{j + 1}"],
                c=z_var,
                marker="o",
                edgecolor="",
                cmap="Spectral_r",
                s=3,
                alpha=0.7,
            )

            lims = [x_max, x_min]
            axs[j][i].plot(lims, lims, "k-", lw=0.3, zorder=0)

        axs[0][i].set_title(f"#(guides) = {i + 1}")

        if i == 0:
            for l_index, l in enumerate(y_features):
                axs[l_index][i].set_ylabel(f"{l}\n{dtype}")

        axs[len(y_features) - 1][i].set_xlabel(f"{x_feature}\n{dtype}")

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(
        f"{rpath}/ky_v11_ks_guides_benchmark_scatter_{dtype}.png", bbox_inches="tight"
    )
    plt.close("all")
