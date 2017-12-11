#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from crispy.benchmark_plot import plot_chromosome
from sklearn.linear_model import LinearRegression, Ridge


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
nexp_df = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

# GDSC
gdsc_kmean = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# MOBEMs
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)
mobems = mobems.loc[[i for i in mobems.index if i.startswith('gain')]]

# Copy-number ratios non-expressed genes
df = pd.read_csv('data/crispr_gdsc_cnratio_nexp.csv')


# - Overlap and Filter
samples = set(gdsc_kmean).intersection(mobems).intersection(nexp_df)

nexp_df = nexp_df[nexp_df.loc[:, samples].sum(1) >= 3]
mobems = mobems[mobems.loc[:, samples].sum(1) >= 3]

genes = set(gdsc_kmean.index).intersection(nexp_df.index)
print(len(samples), len(genes))


# -
y = (gdsc_kmean.loc[genes, samples] * nexp_df.loc[genes, samples]).T
x = mobems[y.index].T

lms = Ridge(normalize=True).fit(x, y)

lm_betas = pd.DataFrame(lms.coef_, index=y.columns, columns=x.columns)

lm_df = lm_betas.unstack().sort_values(ascending=True)
print(lm_df.head(10))


# -
y_feat, x_feat = 'NEUROD2', 'gain:cnaPANCAN301 (CDK12,ERBB2,MED24)'

plot_df = pd.concat([
    nexp_df.loc[y_feat, samples].rename('expression'),
    gdsc_kmean.loc[y_feat, samples].rename('crispr'),
    mobems.loc[x_feat, samples].rename('genomic'),
    ss.loc[samples, 'Cancer Type'].rename('type')
], axis=1).dropna().sort_values('crispr')

sns.boxplot('crispr', 'genomic', data=plot_df, color=bipal_dbgd[0], orient='h', sym='')
sns.stripplot('crispr', 'genomic', data=plot_df, color=bipal_dbgd[0], orient='h', edgecolor='white', linewidth=.1, size=5, jitter=.4)

plt.axvline(0, ls='-', lw=.1, c=bipal_dbgd[0])

plt.xlabel('%s CRISPR/Cas9 bias' % y_feat)
plt.ylabel(x_feat)

plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/collateral_essentiality_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
t = 'Breast Carcinoma'
c = cnexp.loc[ctype_lm[t].argmin()].sort_values().head().index[0]

df = pd.concat([
    ccrispy[c].rename('fc'),
    c_gdsc[c].rename('bias'),
    cnv_abs[c].rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos'),
], axis=1).dropna().query('cnv != -1').sort_values(['chr', 'pos'])
df = df.query("chr == '%s'" % df.loc[g, 'chr'])

df_seg = cnv_seg.query("cellLine == '%s'" % c).query("chr == '%s'" % df.loc[g, 'chr'])

ax = plot_chromosome(df['pos'], df['fc'], df['bias'], seg=df_seg, highlight=[g, 'BRCA1', 'ERBB2'])
# plt.show()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/synthetic_essentiality_chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')
