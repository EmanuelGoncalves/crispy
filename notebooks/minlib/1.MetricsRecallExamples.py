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
from minlib import rpath, dpath, LOG, project_score_sample_map
from sklearn.metrics import recall_score, precision_score


def calculate_avg_gene_fc(counts, library, plasmids=None):
    sample_map = project_score_sample_map()
    plasmids = ["CRISPR_C6596666.sample"] if plasmids is None else plasmids

    fc = counts.loc[library.index]

    fc = fc.norm_rpm().foldchange(plasmids)

    fc = fc.groupby(library["Gene"]).mean()

    fc = fc.groupby(sample_map["model_id"], axis=1).mean()

    return fc


# Import sgRNA counts and efficiency metrics

ky = CRISPRDataSet("Yusa_v1.1")
ky_counts = ky.counts.remove_low_counts(ky.plasmids)


# Minimal library

min_lib = Library.load_library("minimal_top_2.csv.gz")
min_lib = min_lib[min_lib.index.isin(ky_counts.index)]


# Master library

master_lib = (
    Library.load_library("master.csv.gz")
    .query("lib == 'KosukeYusa'")
    .dropna(subset=["Gene"])
)
master_lib = master_lib[master_lib.index.isin(ky_counts.index)]
master_lib = master_lib[master_lib["Gene"].isin(min_lib["Gene"])]


# Gene-level fold-changes

fc_master = calculate_avg_gene_fc(ky_counts, master_lib)
fc_minimal = calculate_avg_gene_fc(ky_counts, min_lib)

fc_dsets = [("master", fc_master), ("minimal", fc_minimal)]


# Overlap

GENES = set(fc_master.index).intersection(fc_minimal.index)
SAMPLES = set(fc_master).intersection(fc_minimal)
LOG.info(f"samples={len(SAMPLES)}; Genes={len(GENES)}")


# Recall

FDR_THRES = 0.05

recall_ess = pd.DataFrame(
    {
        n: {s: QCplot.precision_recall_curve(df[s], fdr_thres=FDR_THRES)[0] for s in SAMPLES}
        for n, df in fc_dsets
    }
)
recall_ess = recall_ess.loc[recall_ess.eval("master - minimal").sort_values().index]


#

sample = "SIDM00530"

plt.figure(figsize=(1.5, 1.5), dpi=600)

for i, (n, df) in enumerate(fc_dsets):
    ap, recall_fdr, precision, recall, thres = QCplot.precision_recall_curve(
        df[sample], fdr_thres=FDR_THRES, return_curve=True
    )

    plt.plot(recall, precision, label=n, color=CrispyPlot.PAL_DBGD[i], lw=.5)

    plt.axhline(1 - FDR_THRES, ls="-", lw=0.5, c=CrispyPlot.PAL_DBGD[3], zorder=0)

    plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")


plt.legend(prop={"size": 5}, frameon=False)
plt.savefig(f"{rpath}/metrics_curve.png", bbox_inches="tight")
plt.close("all")
