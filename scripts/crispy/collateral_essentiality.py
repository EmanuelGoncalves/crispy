#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from sklearn.linear_model import Ridge


# - Imports
# Copy-number ratios
cnv_ratio = pd.read_csv('data/copy_number_ratio.csv')
cnv_ratio_m = pd.pivot_table(cnv_ratio, index='gene', columns='sample', values='ratio').dropna()

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
nexp_df = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

# CRISPR bias
gdsc_kmean = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()


# - Overlap
samples = set(gdsc_kmean).intersection(nexp_df).intersection(cnv_ratio_m)
genes = set(nexp_df.index).intersection(gdsc_kmean.index)
print(len(samples), len(genes))


# - Linear regressions
x = cnv_ratio_m.loc[:, samples].T
x = x.loc[:, (x > 2).sum() >= 3]
x = (x > 2).astype(int)

lm_betas = {}
for g in genes:
    y = gdsc_kmean.loc[g, samples][nexp_df.loc[g, samples] == 1]
    if (len(y) >= 10) and (sum(y.abs() > .5) >= 3):
        lm = Ridge().fit(x.loc[y.index], y)
        lm_betas[g] = dict(zip(x.columns, lm.coef_))

lm_betas = pd.DataFrame(lm_betas)

lm_df = lm_betas.unstack().sort_values(ascending=True).reset_index().rename(columns={'level_0': 'ratio', 'level_1': 'crispr', 0: 'beta'})
print(lm_df.head(60))


# -
y_feat, x_feat = 'OCM2', 'TCF20'

plot_df = pd.concat([
    gdsc_kmean.loc[y_feat, samples][nexp_df.loc[y_feat, samples] == 1].rename('crispr'),
    (cnv_ratio_m.loc[x_feat, samples].rename('ratio') > 2).astype(int),
], axis=1).dropna().sort_values('crispr')

#
sns.boxplot('ratio', 'crispr', data=plot_df, color=bipal_dbgd[0], sym='')
sns.stripplot('ratio', 'crispr', data=plot_df, color=bipal_dbgd[0], edgecolor='white', linewidth=.1, size=3, jitter=.1)

plt.axhline(0, ls='-', lw=.1, c=bipal_dbgd[0])

plt.ylabel('%s CRISPR/Cas9 bias' % y_feat)
plt.xlabel(x_feat)

plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/collateral_essentiality_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

