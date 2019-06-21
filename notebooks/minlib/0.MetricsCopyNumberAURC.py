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
from crispy.QCPlot import QCplot
from crispy.DataImporter import CopyNumber
from crispy.CRISPRData import CRISPRDataSet, ReadCounts
from minlib import rpath, LOG, assemble_corrected_fc, ky_v11_calculate_gene_fc


# Project Score samples acquired with Kosuke_Yusa v1.1 library

ky_v11_data = CRISPRDataSet("Yusa_v1.1")


# Copy-number information

cn = CopyNumber()


# Guide selection metrics

metrics = ["all", "ks_control_min"]


# Calculate original fold-changes

ky_v11_fc_all = (
    ReadCounts(
        pd.read_csv(f"{rpath}/KosukeYusa_v1.1_sgrna_counts_all.csv.gz", index_col=0)
    )
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)
ky_v11_fc_all = ky_v11_calculate_gene_fc(
    ky_v11_fc_all, ky_v11_data.lib.loc[ky_v11_fc_all.index]
)

ky_v11_fc_ks_top2 = (
    ReadCounts(
        pd.read_csv(
            f"{rpath}/KosukeYusa_v1.1_sgrna_counts_ks_control_top2.csv.gz", index_col=0
        )
    )
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)
ky_v11_fc_ks_top2 = ky_v11_calculate_gene_fc(
    ky_v11_fc_ks_top2, ky_v11_data.lib.loc[ky_v11_fc_ks_top2.index]
)

ky_v11_fc_ks_top3 = (
    ReadCounts(
        pd.read_csv(
            f"{rpath}/KosukeYusa_v1.1_sgrna_counts_ks_control_top2.csv.gz", index_col=0
        )
    )
    .norm_rpm()
    .foldchange(ky_v11_data.plasmids)
)
ky_v11_fc_ks_top3 = ky_v11_calculate_gene_fc(
    ky_v11_fc_ks_top3, ky_v11_data.lib.loc[ky_v11_fc_ks_top3.index]
)


# CRISPRcleanR corrected fold-changes

dfolder = f"{rpath}/KosukeYusa_v1.1_crisprcleanr/"
ky_v11_gene_cc = {
    m: {
        n: assemble_corrected_fc(dfolder, m, n, "crisprcleanr", lib=ky_v11_data.lib)
        for n in [2, 3]
    }
    for m in metrics
}


# Crispy corrected fold-changes

dfolder = f"{rpath}/KosukeYusa_v1.1_crispy/"
ky_v11_gene_crispy = {
    m: {
        n: assemble_corrected_fc(dfolder, m, n, "crispy", lib=ky_v11_data.lib)
        for n in [2, 3]
    }
    for m in metrics
}


# Calculate AROC

fc_datasets = [
    ("original", "all", ky_v11_fc_all),
    ("original", "top2", ky_v11_fc_ks_top2),
    ("original", "top3", ky_v11_fc_ks_top3),
    ("ccleanr", "all", ky_v11_gene_cc["all"][2]),
    ("ccleanr", "top2", ky_v11_gene_cc["ks_control_min"][2]),
    ("ccleanr", "top3", ky_v11_gene_cc["ks_control_min"][3]),
    ("crispy", "all", ky_v11_gene_crispy["all"][2]),
    ("crispy", "top2", ky_v11_gene_crispy["ks_control_min"][2]),
    ("crispy", "top3", ky_v11_gene_crispy["ks_control_min"][3]),
]

samples = (
    set(ky_v11_fc_all)
    .intersection(ky_v11_gene_cc["ks_control_min"][2])
    .intersection(ky_v11_gene_crispy["ks_control_min"][3])
)

ky_v11_cn_auc = []
for dtype, n_guides, dtype_df in fc_datasets:
    LOG.info(f"dtype={dtype}; n_guides={n_guides}")

    # Assemble data-frame
    m_cn_df = pd.concat(
        [
            cn.get_data()[samples].unstack().rename("cn"),
            dtype_df[samples].unstack().rename("fc"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    m_cn_df = m_cn_df.reset_index().rename(
        columns={"level_0": "sample", "level_1": "gene"}
    )

    # Calculate AROC for each sample
    m_cn_aucs_df = pd.concat(
        [
            QCplot.recall_curve_discretise(df["fc"], df["cn"], 10)
            .assign(sample=s)
            .assign(dtype=dtype)
            .assign(n_guides=n_guides)
            for s, df in m_cn_df.groupby("sample")
        ]
    )

    ky_v11_cn_auc.append(m_cn_aucs_df)
ky_v11_cn_auc = pd.concat(ky_v11_cn_auc).reset_index(drop=True)
ky_v11_cn_auc.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_cn.xlsx", index=False)
