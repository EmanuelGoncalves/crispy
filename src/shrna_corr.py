#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy_sugar.stats import quantile_gaussianize

# - shRNA
# Project DRIVE
d_drive = pd.read_csv('data/ccle/DRIVE_ATARiS_data.txt', sep='\t')
d_drive.columns = [c.upper() for c in d_drive]
print('Project DRIVE: ', d_drive.shape)

# Project Achilles
d_achilles = pd.read_csv('data/ccle/Achilles_v2.20.2_GeneSolutions.txt', sep='\t', index_col=0).drop('Description', axis=1)
print('Project Achilles: ', d_achilles.shape)


# - CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crispy_fold_change.csv', index_col=0)
crispr = crispr.groupby([i.split('_')[0].replace('sg', '') for i in crispr.index]).mean()
print('CRISPR: ', crispr.shape)

# - Overlap
genes, samples = set(d_drive.index).intersection(d_achilles.index), set(d_drive).intersection(d_achilles)
print('Overlap: ', len(genes), len(samples))


# - Correlation
corr = []
for g in genes:
    df = pd.DataFrame({'drive': d_drive.loc[g, samples], 'achilles': d_achilles.loc[g, samples]}).dropna()

    cor, pval = pearsonr(df['drive'], df['achilles'])

    corr.append({'gene': g, 'cor': cor, 'pval': pval, 'len': df.shape[0]})

    print(g)
corr = pd.DataFrame(corr).sort_values('cor', ascending=False)
print(corr)

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
sns.distplot(corr['cor'], kde=False, bins=30, hist_kws={'color': '#34495e'})
plt.title('Mean Pearson R = %.2f' % corr['cor'].mean())
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/shrna_corr_hist.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Map overlaping cell lines
cmap = pd.read_csv('data/ccle/CCLE_GDSC_cellLineMappoing.csv', index_col=2)

genes = set(d_drive.index).intersection(d_achilles.index).intersection(crispr.index)

samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_drive).intersection(d_achilles)
print('Overlap: ', len(genes), len(samples))

d_drive, d_achilles = d_drive.loc[:, samples], d_achilles.loc[:, samples]
d_drive.columns, d_achilles.columns = list(cmap.loc[d_drive.columns, 'GDSC1000 name']), list(cmap.loc[d_achilles.columns, 'GDSC1000 name'])
crispr = crispr[d_drive.columns]

samples = set(crispr).intersection(d_drive).intersection(d_achilles)
print('Overlap: ', len(genes), len(samples))


# - Correlation CRISPR
c_corr = []
for g in genes:
    df = pd.DataFrame({
        'drive': d_drive.loc[g, samples],
        'achilles': d_achilles.loc[g, samples],
        'crispr': crispr.loc[g, samples]
    }).dropna()

    d_cor, d_pval = pearsonr(df['crispr'], df['drive'])
    a_cor, a_pval = pearsonr(df['crispr'], df['achilles'])

    c_corr.append({
        'gene': g, 'len': df.shape[0],
        'cor_drive': d_cor, 'pval_drive': d_pval,
        'cor_achilles': a_cor, 'pval_achilles': a_pval,
    })

    print(g)
c_corr = pd.DataFrame(c_corr).sort_values('cor_drive', ascending=False)
print(c_corr)
print('Corr DRIVE: ', c_corr['cor_drive'].mean(), 'Achilles: ', c_corr['cor_achilles'].mean())

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
sns.distplot(c_corr['cor_drive'], kde=False, bins=50, color='#37454B', label='DRIVE')
sns.distplot(c_corr['cor_achilles'], kde=False, bins=50, color='#F2C500', label='Achilles')
plt.xlabel('Pearson between shRNA and CRISPR')
plt.title('Mean Pearson: DRIVE=%.2f, Achilles=%.2f' % (c_corr['cor_drive'].mean(), c_corr['cor_achilles'].mean()))
plt.legend()
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/shrna_crispr_corr_hist.png', bbox_inches='tight', dpi=600)
plt.close('all')
