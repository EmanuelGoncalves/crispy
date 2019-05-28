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
from crispy.CRISPRData import CRISPRDataSet
from C50K import dpath, rpath, guides_aroc_benchmark


# Import sgRNA counts and efficiency metrics

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_metrics = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx").dropna()


# Benchmark sgRNA: Essential/Non-essential AROC

nguides_thres = 5

ky_v11_arocs = guides_aroc_benchmark(
    ky_v11_metrics,
    ky_v11_count,
    nguides_thres=nguides_thres,
    guide_set=set(ky_v11_metrics.index),
    rand_iterations=10,
)


# Export

ky_v11_arocs.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_aroc.xlsx", index=False)
