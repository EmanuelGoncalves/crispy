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
from C50K import rpath, LOG
from crispy.CRISPRData import CRISPRDataSet


def replicates_correlation(sgrna_data, guides=None):
    # Subset guides
    if guides is not None:
        df = sgrna_data.counts.reindex(guides)

    else:
        df = sgrna_data.counts.copy()

    # Calculate fold-changes
    df = df.remove_low_counts(sgrna_data.plasmids)
    df = df.norm_rpm().foldchange(sgrna_data.plasmids)

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

ky_v11_data = CRISPRDataSet("Yusa_v1.1")


# sgRNA efficiency metrics

metrics = [
    "ks_control",
    "ks_control_min",
    "jacks_min",
    "doenchroot",
    "doenchroot_min",
    "median_fc",
    "Gene",
]

rep_corrs = []
for n_guides in [2, 3]:
    for m in metrics:
        LOG.info(f"metric={m}; sgRNA={n_guides}")

        if m != "Gene":
            m_counts = pd.read_csv(
                f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{m}_top{n_guides}.csv.gz",
                index_col=0,
            )

        else:
            m_counts = pd.read_excel(
                f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0
            )[metrics].dropna()

        m_rep_corr = replicates_correlation(ky_v11_data, guides=list(m_counts.index))

        rep_corrs.append(
            m_rep_corr.assign(metric="all" if m == "Gene" else m).assign(
                n_guides=n_guides
            )
        )

rep_corrs = pd.concat(rep_corrs)

rep_corrs.to_csv(
    f"{rpath}/KosukeYusa_v1.1_benchmark_rep_correlation.csv.gz",
    compression="gzip",
)
