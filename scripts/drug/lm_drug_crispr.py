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

t_meas = []
# gene, drug = 'TP53', (1047, 'Nutlin-3a (-)', 'RS')
for gene_idx, drug_idx in zip(*(range(0, 1000), range(0, d_response.shape[0]))):
    print(gene_idx, drug_idx)

    time_start = time.time()

    # Build data-frames
    yy = d_response.iloc[drug_idx].values
    xx = crispr.iloc[[gene_idx]].T.values

    res = lr(xx, yy)

    time_lm = time.time()

    #
    t_meas.append(('lm', time_lm - time_start))

t_meas = pd.DataFrame(t_meas, columns=['func', 'time'])
print(t_meas.groupby('func').median())
print('Total run time (0.5M tests)', t_meas.groupby('func').median() * 5e5 / 60)
# sns.boxplot('func', 'time', data=t_meas, notch=True)
