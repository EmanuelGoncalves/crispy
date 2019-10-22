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
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from minlib.Utils import project_score_sample_map
from crispy.CRISPRData import CRISPRDataSet, Library
from sklearn.metrics import recall_score, precision_score


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


def guides_recall_benchmark(
    metrics, sgrna_counts, dataset, smap, nguides_thres=None, jacks_thres=1., fpr_thres=0.01
):
    nguides_thres = [1, 2, 3, 4, 5, 100] if nguides_thres is None else nguides_thres

    # Define set of guides
    LOG.info(f"#(sgRNAs)={metrics.shape[0]}")

    # AROC scores
    scores = []

    for m in ["KS", "JACKS_min", "combined"]:
        LOG.info(f"Metric = {m}")

        for n in nguides_thres:
            # Metric top guides
            if m == "combined":
                metric_topn = (
                    metrics.query(f"JACKS_min < {jacks_thres}")
                    .sort_values("KS", ascending=False)
                    .groupby("Approved_Symbol")
                    .head(n=n)
                )

            else:
                metric_topn = (
                    metrics.sort_values(m, ascending=(m == "JACKS_min")).groupby("Approved_Symbol").head(n=n)
                )

            # Calculate fold-changes on subset
            metric_fc = sgrna_counts.loc[metric_topn.index]
            metric_fc = metric_fc.norm_rpm().foldchange(dataset.plasmids)
            metric_fc = metric_fc.groupby(metrics["Approved_Symbol"]).mean()
            metric_fc = metric_fc.groupby(smap["model_id"], axis=1).mean()

            # Binarise fold-changes
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
                        ess_aroc=QCplot.aroc_threshold(
                            metric_fc.loc[genes, s], fpr_thres=.2
                        )[0],
                        recall=recall_score(
                            ky_bin.loc[genes, s], metric_bin.loc[genes, s]
                        ),
                        precision=precision_score(
                            ky_bin.loc[genes, s], metric_bin.loc[genes, s]
                        ),
                    )
                    for s in metric_bin
                ]
            )

            # Store
            scores.append(metric_recalls)

    scores = pd.concat(scores)

    return scores


# Project Score KY v1.1
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# Master library (KosukeYusa v1.1 + Avana + Brunello)
#

master_lib = (
    Library.load_library("MasterLib_v1.csv.gz", set_index=True)
    .query("Library == 'KosukeYusa'")
    .dropna(subset=["Approved_Symbol", "KS", "JACKS"])
)
master_lib["JACKS_min"] = abs(master_lib["JACKS"] - 1)
master_lib = master_lib[master_lib.index.isin(ky_counts.index)]


# Project Score KY v1.1: Fold-changes
#

FDR_THRES = 0.01

ky_sgrna_fc = ky_counts.loc[master_lib.index].norm_rpm().foldchange(ky.plasmids)

ky_fc = (
    ky_sgrna_fc.groupby(master_lib["Approved_Symbol"])
    .mean()
    .groupby(ky_smap["model_id"], axis=1)
    .mean()
)

ky_thres = [QCplot.aroc_threshold(ky_fc[s], fpr_thres=FDR_THRES)[1] for s in ky_fc]
ky_bin = (ky_fc < ky_thres).astype(int)


# Benchmark sgRNA: Essential/Non-essential AROC
#

metrics_recall = guides_recall_benchmark(master_lib, ky_counts, ky, ky_smap, fpr_thres=FDR_THRES, jacks_thres=1.)
metrics_recall.to_excel(f"{RPATH}/KosukeYusa_v1.1_benchmark_recall.xlsx", index=False)


# Boxplot the benchmark scores
#

n_cols = len(set(metrics_recall["metric"]))
n_rows = len(set(metrics_recall["nguides"]))

order = ["KS", "JACKS_min", "combined"]
order_col = ["ess_aroc", "precision", "recall"]

pal = dict(
    KS=CrispyPlot.PAL_DBGD[0],
    JACKS_min=CrispyPlot.PAL_DBGD[2],
    combined=CrispyPlot.PAL_DBGD[1],
)

f, axs = plt.subplots(
    n_rows, n_cols, sharey="all", sharex="col", figsize=(2. * n_cols, .75 * n_rows)
)

for i, nguides in enumerate(set(metrics_recall["nguides"])):
    for j, metric in enumerate(order_col):
        ax = axs[i][j]

        df = pd.pivot_table(
            metrics_recall.query(f"nguides == '{nguides}'").drop(columns=["nguides"]),
            index="sample", columns="metric"
        )[metric]

        for z, v in enumerate(order):
            ax.boxplot(
                df[v],
                positions=[z],
                vert=False,
                showcaps=False,
                patch_artist=True,
                boxprops=dict(linewidth=0.3, facecolor=pal[v]),
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
                widths=.8,
            )

        ax.set_yticklabels(order)

        ax.set_xlim(right=1.05)

        ax.set_xlabel(metric if i == (n_rows - 1) else "")
        ax.set_ylabel(f"{'All' if nguides == 100 else nguides} sgRNAs" if j == 0 else None)

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig(f"{RPATH}/MetricsRecall_boxplots.pdf", bbox_inches="tight", transparent=True)
plt.close("all")
