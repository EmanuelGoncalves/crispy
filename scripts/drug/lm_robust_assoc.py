#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
from crispy.regression.linear import lr
from statsmodels.stats.multitest import multipletests


# - Imports
# Crispr ~ Drug associations
lm_df = pd.read_csv('data/drug/lm_drug_crispr.csv')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

# Growth rate
growth = pd.read_csv('data/gdsc/growth_rate.csv', index_col=0)

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()
crispr = crispr.subtract(crispr.mean()).divide(crispr.std())

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# MOBEMs
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)
mobems.index.name = 'genomic'


# - Filter
mobems = mobems[mobems.sum(1) >= 5]


# - Overlap
samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index).intersection(mobems))
d_response, crispr, mobems = d_response[samples], crispr[samples], mobems[samples]
print('Samples: %d' % len(samples))


# - Rosbut pharmacogenomic associations
ppairs = lm_df.query('lr_fdr < 0.05')[['GENES', 'DRUG_ID', 'DRUG_NAME', 'VERSION']].values
print('<Drug, CRISPR>: %d; Genomics: %d' % (len(ppairs), len(mobems.index)))

# Build matrices
xs, ws = mobems[samples].T, pd.concat([pd.get_dummies(ss[['Cancer Type']]), growth['NC1_ratio_mean']], axis=1).loc[samples]

# Linear regressions
lm_res = []
for gene, drug_id, drug_name, drug_screen in ppairs[:3]:
    # Linear regressions of drug and crispr
    lm_res_crispr = lr(xs, crispr.loc[[gene], samples].T, ws)
    lm_res_drug = lr(xs, d_response.loc[[(drug_id, drug_name, drug_screen)], samples].T, ws)

    # Melt result matrices
    lm_res_crispr_df = pd.concat([lm_res_crispr[i].unstack().rename(i) for i in lm_res_crispr], axis=1).reset_index().set_index('genomic').add_prefix('crispr_')
    lm_res_drug_df = pd.concat([lm_res_drug[i].unstack().rename(i) for i in lm_res_drug], axis=1).reset_index().set_index('genomic').add_prefix('drug_')

    # Concat data-frames
    lm_res.append(pd.concat([lm_res_crispr_df, lm_res_drug_df], axis=1).reset_index())

# Merge results
lm_res = pd.concat(lm_res)

lm_res = lm_res.assign(drug_lr_fdr=multipletests(lm_res['drug_lr_pval'], method='fdr_bh')[1])
lm_res = lm_res.assign(crispr_lr_fdr=multipletests(lm_res['crispr_lr_pval'], method='fdr_bh')[1])

# Export
lm_res.sort_values('lr_fdr').to_csv('data/drug/lm_drug_crispr_genomic.csv', index=False)
print(lm_res.sort_values('crispr_lr_fdr'))
