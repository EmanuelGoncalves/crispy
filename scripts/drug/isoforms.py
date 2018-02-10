#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scripts.drug as dc
from scripts.drug.lm_drug_crispr import lm_drug_crispr


if __name__ == '__main__':
    # - Imports
    # Drug target
    d_targets = dc.drug_targets()

    # Samplesheet
    ss = pd.read_csv(dc.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv(dc.GROWTHRATE_FILE, index_col=0)

    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv(dc.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    crispr_scaled = dc.scale_crispr(crispr)

    # Drug response
    d_response = pd.read_csv(dc.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index))
    d_response, crispr = d_response[samples], crispr[samples]
    print('Samples: %d' % len(samples))

    # - Transform and filter
    d_response = dc.filter_drug_response(d_response)

    # - Covariates
    covariates = pd.concat([
        pd.get_dummies(ss[['Cancer Type']]),
        growth['growth_rate_median']
    ], axis=1).loc[samples]
    covariates = covariates.loc[:, covariates.sum() != 0]

    # -
    isoforms = ['AKT1', 'AKT2', 'AKT3']

    # - Linear regression: drug ~ crispr + tissue
    lm_res_df = lm_drug_crispr(crispr_scaled.loc[isoforms, samples].T, d_response[samples].T, covariates.loc[samples])
    print(lm_res_df[(lm_res_df['beta'].abs() > .5) & (lm_res_df['lr_fdr'] < 0.05)].sort_values('lr_fdr'))
