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
from minlib import rpath, guides_aroc_benchmark, LOG


# Import sgRNA counts and efficiency metrics

ky_v11_count = CRISPRDataSet("Yusa_v1.1")
ky_v11_metrics = pd.read_excel(
    f"{rpath}/KosukeYusa_v1.1_sgRNA_metrics.xlsx", index_col=0
)


# Selected metrics + Gene annotation + drop NaNs

metrics = [
    "ks_control",
    "ks_control_min",
    "jacks_min",
    "doenchroot",
    "doenchroot_min",
    "median_fc",
    "Gene",
]
ky_v11_metrics = ky_v11_metrics[metrics].dropna()


# Benchmark sgRNA: Essential/Non-essential AROC

nguides_thres = 5

ky_v11_arocs = guides_aroc_benchmark(
    ky_v11_metrics,
    ky_v11_count,
    nguides_thres=nguides_thres,
    guide_set=set(ky_v11_metrics.index),
    rand_iterations=10,
)

ky_v11_arocs.to_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_aroc.xlsx", index=False)


# Export counts of Top 2/3 sgRNA of each metric

for n_guides in [2, 3]:
    for m in metrics:
        if m != "Gene":
            LOG.info(f"metric={m}; sgRNA={n_guides}")

            ky_v11_count_m = ky_v11_count.counts.loc[
                ky_v11_metrics.sort_values(m).groupby("Gene").head(n=n_guides).index
            ]

            ky_v11_count_m.to_csv(
                f"{rpath}/KosukeYusa_v1.1_sgrna_counts_{m}_top{n_guides}.csv.gz",
                compression="gzip",
            )
