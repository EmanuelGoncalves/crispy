#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pickle
import numpy as np
import pandas as pd
import itertools as it
import seaborn as sns
import matplotlib.pyplot as plt
from limix.plot import qqplot
from limix.stats import qvalues
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from limix.qtl import qtl_test_lm, qtl_test_lmm
from sklearn.preprocessing import StandardScaler
from limix_core.util.preprocess import gaussianize
from sklearn.metrics.regression import explained_variance_score


# - Import data-sets
# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()
crispr = crispr.subtract(crispr.mean()).divide(crispr.std())

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)
gexp = gexp.subtract(gexp.mean()).divide(gexp.std())

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])


# - Filter + Overlap
# Filter
samples = list(set(crispr).intersection(gexp).intersection(ss.index))
crispr, gexp = crispr[samples], gexp[samples]

# CRISPR
crispr = crispr[(crispr < crispr.loc[essential].mean().mean()).sum(1) < (crispr.shape[1] * .5)]
crispr = crispr[(crispr.abs() >= 2).sum(1) >= 5]

# Gexp
gexp = gexp[(gexp < -1).sum(1) < (gexp.shape[1] * 0.5)]
gexp = gexp[(gexp.abs() >= 1.5).sum(1) >= 5]
print(crispr.shape, gexp.shape)


# -
# Build matrices
y, x = gexp[samples].T, crispr[samples].T

# Standardize xs
x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)

# Covariates
covs = pd.get_dummies(ss.loc[samples, 'Cancer Type']).astype(float)

# Linear regression
lmm = qtl_test_lm(pheno=y.values, snps=x.values, covs=covs.values)

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
lmm_df = lmm_df.assign(qvalue=qvalues(lmm_df['pvalue'].values)).sort_values('qvalue')
print(lmm_df.query('qvalue < 0.05'))


qqplot(lmm.getPv().ravel())
plt.show()

lmm_df.query('qvalue < 0.05').to_csv('data/sl/gexp_crispr.csv', index=False)


# - Plot
# Fit function
def fun(x, a, b, c):
    return a + b * x + c * (x ** 2)


sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
gs, pos = GridSpec(10, 5, hspace=.5, wspace=.5), 0
# gx, gy = 'FAM50B', 'FAM50A'
for beta, gx, gy, pval, qval in lmm_df.head(50).values:
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
    # ax.set_title('y = {:.2f} + {:.2f} * x + {:.2f} * x**2'.format(*popt))
    ax.set_title('B=%.2f; FDR=%.2e' % (beta, qval))

    pos += 1

plt.gcf().set_size_inches(2.5 * 5, 2.5 * 10)
plt.savefig('reports/sl/nlfit_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')
