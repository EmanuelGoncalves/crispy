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
from crispy.DataImporter import Mobem
from crispy.CRISPRData import CRISPRDataSet, ReadCounts, Library
from minlib import rpath, dpath, LOG, project_score_sample_map, lm_associations


# Genomic data

mobem = Mobem()


# Project Score samples acquired with Kosuke_Yusa v1.1 library

ky = CRISPRDataSet("Yusa_v1.1")

s_map = project_score_sample_map()


# sgRNAs lib

mlib = (
    Library.load_library("master.csv.gz", set_index=False)
    .query("lib == 'KosukeYusa'")
    .dropna(subset=["Gene"])
    .set_index("sgRNA_ID")
)
mlib = mlib[mlib.index.isin(ky.counts.index)]


# Strongly Selective Dependencies (SSDs)

genes = list(pd.read_excel(f"{dpath}/ssd_genes.xls")["CancerGenes"])


# Biomarker benchmark

DTYPES = ["all", "minimal_top_2", "minimal_top_3"]

lm_assocs = []
for dtype in DTYPES:
    # Read counts
    if dtype == "all":
        counts = ky.counts.copy()

    else:
        counts = ReadCounts(
            pd.read_csv(
                f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{dtype}.csv.gz", index_col=0
            )
        )

    # sgRNA fold-changes
    fc = counts.remove_low_counts(ky.plasmids).norm_rpm().foldchange(ky.plasmids)

    # gene fold-changes
    fc_gene = fc.groupby(mlib.loc[fc.index, "Gene"]).mean()

    # gene average fold-changes
    fc_gene_avg = fc_gene.groupby(s_map["model_id"], axis=1).mean()

    samples = list(set(fc_gene_avg).intersection(mobem.get_data()))

    m_mobem = mobem.get_data()[samples].T
    m_mobem = m_mobem.loc[:, m_mobem.sum() > 3]

    LOG.info(
        f"metric={dtype}; samples={len(samples)}; mobem={m_mobem.shape[1]}"
    )

    lm_assocs.append(
        lm_associations(fc_gene_avg.loc[genes, samples].T, m_mobem.loc[samples])
        .assign(metric=dtype)
    )

lm_assocs = pd.concat(lm_assocs)
lm_assocs.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_biomarker.xlsx", index=False)
