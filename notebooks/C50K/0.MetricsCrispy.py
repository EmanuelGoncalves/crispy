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
from C50K import rpath, dpath, LOG
from crispy.CopyNumberCorrection import Crispy
from crispy.DataImporter import CopyNumberSegmentation
from crispy.CRISPRData import CRISPRDataSet, ReadCounts


# Import Project Score samples acquired with Kosuke_Yusa v1.1 library

ky_v11_data = CRISPRDataSet("Yusa_v1.1")


# Project Score sample map

smap = pd.read_csv(
    f"{dpath}/crispr_manifests/project_score_sample_map_KosukeYusa_v1.1.csv.gz",
    compression="gzip",
)


# Copy-number segmentation

copy_number = CopyNumberSegmentation().get_data()


# CRISPR-Cas9 library

library = ky_v11_data.lib.reset_index().rename(
    columns={"sgRNA_ID": "sgrna", "Gene": "gene"}
)


# Samples overlap

samples = list(set(smap["model_id"]).intersection(set(copy_number["model_id"])))


# Execute Crispy

metrics = [
    "ks_control",
    "ks_control_min",
    "jacks_min",
    "doenchroot",
    "doenchroot_min",
    "median_fc",
    "Gene",
]

for n_guides in [2, 3]:
    for m in ["ks_control_min", "Gene"]:

        # Import counts
        if m == "Gene":
            if n_guides != 2:
                continue

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

        for s in samples:
            LOG.info(f"metric={m}; sgRNA={n_guides}; sample={s}")

            # Calculate fold-change
            m_fc = m_counts.norm_rpm().foldchange(ky_v11_data.plasmids)
            m_fc = m_fc[smap.query(f"model_id == '{s}'")["s_lib"]].mean(1)

            # Crispy correction
            m_cy = Crispy(
                sgrna_fc=m_fc,
                library=library,
                copy_number=copy_number.query(f"model_id == '{s}'"),
            )
            m_cy = m_cy.correct()

            # Export
            m_cy.to_csv(
                f"{rpath}/KosukeYusa_v1.1_crispy/{s}_{'all' if m == 'Gene' else m}_top{n_guides}_corrected_fc.csv.gz",
                index=False,
                compression="gzip",
            )
