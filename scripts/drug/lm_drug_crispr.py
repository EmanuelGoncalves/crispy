#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
from crispy.regression.linear import lr
from sklearn.linear_model import LinearRegression

# - Imports
# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type', 'Microsatellite'])

# Drug samplesheet
ds = pd.read_csv('data/gdsc/drug_samplesheet.csv', index_col=0)

# Growth rate
growth = pd.read_csv('data/gdsc/growth_rate.csv', index_col=0)


# - Overlap
samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index))
d_response, crispr = d_response[samples], crispr[samples]
print('Samples: %d' % len(samples))


# - Filter
# Drug response
d_response = d_response[d_response.count(1) > d_response.shape[1] * .85]
d_response = d_response.loc[(d_response < d_response.mean().mean()).sum(1) >= 5]
d_response = d_response.loc[[iqr(values, nan_policy='omit') > 1 for idx, values in d_response.iterrows()]]

# CRISPR
crispr = crispr[(crispr < crispr.loc[essential].mean().mean()).sum(1) < (crispr.shape[1] * .5)]
crispr = crispr[(crispr.abs() >= 2).sum(1) >= 5]


# - lmm: drug ~ crispr + tissue
print('CRISPR genes: %d, Drug: %d' % (len(set(crispr.index)), len(set(d_response.index))))

xs, ys = crispr.sample(10).T, d_response.T

time_start = time.time()

lm_res = lr(xs, ys)

time_elapsed = time.time() - time_start

time_per_run = time_elapsed / (xs.shape[1] * ys.shape[1])

print(time_per_run)
print('Total run time (0.5M tests): %.2f mins' % (time_per_run * 5e5 / 60))


# -
lm_res_df = pd.concat([lm_res[i].unstack().rename(i) for i in lm_res], axis=1).reset_index()
print(lm_res_df.sort_values('f_pval'))

