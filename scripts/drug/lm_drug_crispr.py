#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import iqr
from crispy.utils import qnorm
from crispy.regression.linear import lr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


CRISPR_GENES_FILE = 'data/gdsc/crispr/_00_Genes_for_panCancer_assocStudies.txt'


def crispr_genes(file=None, samples_thres=5, type_thres=3):
    file = CRISPR_GENES_FILE if file is None else file

    c_genes = pd.read_csv(file, sep='\t', index_col=0)

    c_genes = c_genes[c_genes.drop('n. vulnerable cell lines', axis=1).sum(1) <= type_thres]

    c_genes = c_genes[c_genes['n. vulnerable cell lines'] >= samples_thres]

    return set(c_genes.index)


def filter_drug_response(df, percentage_measurements=0.85, ic50_samples=5, iqr_thres=1):
    df = df[df.count(1) > df.shape[1] * percentage_measurements]

    df = df.loc[(df < df.mean().mean()).sum(1) >= ic50_samples]

    df = df.loc[[iqr(values, nan_policy='omit') > iqr_thres for idx, values in df.iterrows()]]

    return df


def lm_drug_crispr(xs, ys, ws):
    print('CRISPR genes: %d, Drug: %d' % (len(set(xs.index)), len(set(ys.index))))

    # Standardize xs
    xs = pd.DataFrame(StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns)

    # Regression
    res = lr(xs, ys, ws)

    # - Export results
    res_df = pd.concat([res[i].unstack().rename(i) for i in res], axis=1).reset_index()

    res_df = res_df.assign(f_fdr=multipletests(res_df['f_pval'], method='fdr_bh')[1])
    res_df = res_df.assign(lr_fdr=multipletests(res_df['lr_pval'], method='fdr_bh')[1])

    return lm_res_df


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv('data/gdsc/growth/growth_rate.csv', index_col=0)

    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv('data/gdsc/crispr/corrected_logFCs.tsv', index_col=0, sep='\t').dropna()

    # Drug response
    d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index))
    d_response, crispr = d_response[samples], crispr[samples]
    print('Samples: %d' % len(samples))

    # - Transform and filter
    d_response = filter_drug_response(d_response)

    crispr_qnorm = pd.DataFrame({c: qnorm(crispr[c]) for c in crispr}, index=crispr.index)
    crispr_qnorm = crispr_qnorm.reindex(crispr_genes()).dropna()

    # - Covariates
    ws = pd.concat([
        pd.get_dummies(ss[['Cancer Type']]),
        growth['growth_rate_median']
    ], axis=1).loc[samples]

    # - Linear regression: drug ~ crispr + tissue
    lm_res_df = lm_drug_crispr(crispr_qnorm[samples].T, d_response[samples].T, ws[samples])

    lm_res_df.sort_values('lr_fdr').to_csv('data/drug/lm_drug_crispr.csv', index=False)
    print(lm_res_df.query('lr_fdr < 0.05').sort_values('lr_fdr'))
