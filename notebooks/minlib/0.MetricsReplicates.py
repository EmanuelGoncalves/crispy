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
from minlib import rpath, LOG
from crispy.CRISPRData import CRISPRDataSet, Library


def replicates_correlation(sgrna_data, guides=None, plasmids=None):
    # Subset guides
    if guides is not None:
        df = sgrna_data.reindex(guides).copy()

    else:
        df = sgrna_data.copy()

    plasmids = ["CRISPR_C6596666.sample"] if plasmids is None else plasmids

    # Calculate fold-changes
    df = df.remove_low_counts(plasmids).norm_rpm().foldchange(plasmids)

    # Remove replicate tags
    df.columns = [c.split("_")[0] for c in df]

    # Sample correlation
    df_corr = df.corr()
    df_corr = df_corr.where(np.triu(np.ones(df_corr.shape), 1).astype(np.bool))
    df_corr = df_corr.unstack().dropna().reset_index()
    df_corr.columns = ["sample_1", "sample_2", "corr"]
    df_corr["replicate"] = (
        (df_corr["sample_1"] == df_corr["sample_2"])
        .replace({True: "Yes", False: "No"})
        .values
    )

    return df_corr


# Import Project Score samples acquired with Kosuke_Yusa v1.1 library

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


# sgRNA efficiency metrics

rep_corrs = []
for n, library in [("all", master_lib), ("minimal_top_2", min_lib)]:
    LOG.info(f"metric={n}; sgRNA={library.shape[0]}")

    m_rep_corr = replicates_correlation(ky_counts, guides=list(library.index)).assign(metric=n)

    rep_corrs.append(m_rep_corr)

rep_corrs = pd.concat(rep_corrs)

rep_corrs.to_csv(
    f"{rpath}/KosukeYusa_v1.1_benchmark_rep_correlation.csv.gz",
    compression="gzip",
)
