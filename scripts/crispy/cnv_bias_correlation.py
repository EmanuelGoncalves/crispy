#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd

# - Imports
# GDSC
c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Copy-number
cnv = pd.read_csv('data/crispy_gene_copy_number_snp.csv', index_col=0)

# Copy-number
ratios = pd.read_csv('data/crispy_gene_copy_number_ratio_snp.csv', index_col=0)


# - Overlap
genes, samples = list(set(c_gdsc.index).intersection(cnv.index).intersection(ratios.index)), list(set(c_gdsc).intersection(cnv).intersection(cnv))
print(len(genes), len(samples))


# - Correlation
plot_df = pd.concat([
    c_gdsc.loc[genes, samples].T.corrwith(cnv.loc[genes, samples].T).rename('cnv'),
    c_gdsc.loc[genes, samples].T.corrwith(ratios.loc[genes, samples].T).rename('ratio'),
], axis=1)

# Histogram
sns.distplot(plot_df['cnv'], label='Copy-number', color=bipal_dbgd[0], kde_kws={'lw': 1.})
sns.distplot(plot_df['ratio'], label='Ratio', color=bipal_dbgd[1], kde_kws={'lw': 1.})

plt.title('Copy-number correlation with CRISPR bias')
plt.xlabel("Pearson's r")
plt.ylabel('Density')
plt.legend(loc='upper left')

plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/crispy/copy_number_bias_corr_histograms.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Cumulative
ax = plt.gca()

_, _, patches = ax.hist(plot_df['cnv'], plot_df.shape[0], cumulative=True, normed=1, histtype='step', label='Copy-number', color=bipal_dbgd[0], lw=1.)
patches[0].set_xy(patches[0].get_xy()[:-1])

_, _, patches = ax.hist(plot_df['ratio'], plot_df.shape[0], cumulative=True, normed=1, histtype='step', label='Ratio', color=bipal_dbgd[1], lw=1.)
patches[0].set_xy(patches[0].get_xy()[:-1])

ax.axvline(0, lw=.1, c=bipal_dbgd[0], alpha=.5, ls='--')

ax.set_title('Copy-number correlation with CRISPR bias')
ax.set_xlabel("Pearson's r")
ax.set_ylabel('Cumulative distribution')

ax.legend(loc='upper left')

plt.gcf().set_size_inches(2.5, 2)
plt.savefig('reports/crispy/copy_number_bias_corr_cumulative.png', bbox_inches='tight', dpi=600)
plt.close('all')
