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
from crispy.CRISPRData import CRISPRDataSet, ReadCounts
from C50K import rpath, dpath, LOG, ky_v11_calculate_gene_fc, lm_associations


# Import genomic data

mobem = Mobem()


# Import Project Score samples acquired with Kosuke_Yusa v1.1 library

ky_v11_data = CRISPRDataSet("Yusa_v1.1")


# Strongly Selective Dependencies (SSDs)

genes = list(pd.read_excel(f"{dpath}/ssd_genes.xls")["CancerGenes"])


# Biomarker benchmark

metrics = [
    "ks_control",
    "ks_control_min",
    "jacks_min",
    "doenchroot",
    "doenchroot_min",
    "median_fc",
    "Gene",
]

lm_assocs = []
for n_guides in [2, 3]:
    for m in metrics:
        if m == "Gene":
            m_counts = pd.read_excel(
                f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0
            )[metrics].dropna()
            m_counts = ky_v11_data.counts.loc[m_counts.index]

        else:
            m_counts = ReadCounts(
                pd.read_csv(
                    f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{m}_top{n_guides}.csv.gz",
                    index_col=0,
                )
            )

        m_fc = m_counts.norm_rpm().foldchange(ky_v11_data.plasmids)

        m_fc_gene = ky_v11_calculate_gene_fc(m_fc, ky_v11_data.lib.loc[m_counts.index])

        samples = list(set(m_fc_gene).intersection(mobem.get_data()))

        m_mobem = mobem.get_data()[samples].T
        m_mobem = m_mobem.loc[:, m_mobem.sum() > 3]

        LOG.info(
            f"metric={m}; sgRNA={n_guides}; samples={len(samples)}; mobem={m_mobem.shape[1]}"
        )

        lm_assocs.append(
            lm_associations(m_fc_gene.loc[genes, samples].T, m_mobem.loc[samples])
            .assign(metric="all" if m == "Gene" else m)
            .assign(n_guides=n_guides)
        )

lm_assocs = pd.concat(lm_assocs)
lm_assocs.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_biomarker.xlsx", index=False)
