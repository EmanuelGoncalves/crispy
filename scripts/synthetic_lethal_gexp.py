#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pickle
import numpy as np
import pandas as pd
import itertools as it
import seaborn as sns
import matplotlib.pyplot as plt
from limix.stats import qvalues
from limix.qtl import qtl_test_lmm
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from limix_core.util.preprocess import gaussianize
from sklearn.metrics.regression import explained_variance_score


# - Import data-sets
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

# sgRNA library
sglib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['GENES'])

# CRISPR
crispr = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)

# - Overlap
samples = list(set(crispr).intersection(gexp))
crispr, gexp = crispr[samples], gexp[samples]
print(crispr.shape, gexp.shape)


# - Filter
# CRISPR
crispr = crispr[(crispr.abs() >= 1).sum(1) >= 5]
crispr = crispr[(crispr < -1).sum(1) < (crispr.shape[1] * .9)]
crispr = crispr.drop(essential, axis=0, errors='ignore')

# Gexp
iqr_l, iqr_u = gexp.unstack().quantile(0.1), gexp.unstack().quantile(0.9)
gexp = gexp.loc[((gexp < iqr_l) | (gexp > iqr_u)).sum(1) >= 10]
print(crispr.shape, gexp.shape)


# -
# Covariates
covars = pd.get_dummies(ss.loc[samples, ['Cancer Type', 'SCREEN MEDIUM', 'Microsatellite']])

# Build matrices
y, x, k = crispr[samples].T, gexp[samples].T, covars.dot(covars.T).astype(float)

# Normalise
x = pd.DataFrame(gaussianize(x.values), index=x.index, columns=x.columns)

# Linear regression
lmm = qtl_test_lmm(pheno=y.values, snps=x.values, K=k.values)

# Extract info
c_gene, g_gene = zip(*list(it.product(y.columns, x.columns)))

# Build pandas data-frame
lmm_df = pd.DataFrame({
    'crispr': c_gene,
    'gexp': g_gene,
    'beta': lmm.getBetaSNP().ravel(),
    'pvalue': lmm.getPv().ravel(),
})
lmm_df = lmm_df[lmm_df['crispr'] != lmm_df['gexp']]
lmm_df = lmm_df.assign(qvalue=qvalues(lmm_df['pvalue'].values))
print(lmm_df.head(50))


# - Plot
# Fit function
def fun(x, a, b, c):
    return a + b * x + c * (x ** 2)


gpairs = [(x, y) for x, y in lmm_df.head(50)[['gexp', 'crispr']].values]

sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
gs, pos = GridSpec(10, 5, hspace=.5, wspace=.5), 0
# gx, gy = 'FAM50B', 'FAM50A'
for gx, gy in gpairs:
    ax = plt.subplot(gs[pos])

    # Get measurements
    x, y = gexp.loc[gx, samples], crispr.loc[gy, samples]

    # Fit curve
    popt, pcov = curve_fit(fun, x, y, method='trf')
    y_ = fun(x, *popt)

    # Variance explained
    vexp = explained_variance_score(y, y_)

    # Plot
    plot_df = pd.DataFrame({
        'x': x, 'y': y, 'y_': y_
    }).sort_values('x')

    ax.scatter(plot_df['x'], plot_df['y'], c='#37454B', alpha=.8, edgecolors='white', linewidths=.3, s=20, label='measurements')
    ax.plot(plot_df['x'], plot_df['y_'], ls='--', c='#F2C500', label='fit')

    ax.axhline(0, lw=.3, ls='-', color='gray', alpha=.5)

    ax.set_xlabel('%s gene-expression' % gx)
    ax.set_ylabel('%s CRISPR' % gy)
    ax.set_title('y = {:.2f} + {:.2f} * x + {:.2f} * x**2'.format(*popt))

    pos += 1

plt.gcf().set_size_inches(2.5 * 5, 2.5 * 10)
plt.savefig('reports/nlfit_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')
