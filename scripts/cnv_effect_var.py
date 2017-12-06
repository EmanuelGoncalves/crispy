#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from matplotlib.colors import rgb2hex
from crispy.benchmark_plot import plot_cnv_rank


# - Imports
# Ploidy
ploidy = pd.read_csv('data/gdsc/cell_lines_project_ploidy.csv', index_col=0)['Average Ploidy']

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

# GDSC
c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(c_gdsc))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[1]))


# - Overlap
genes, samples = set(c_gdsc.index).intersection(cnv_abs.index), set(c_gdsc).intersection(cnv_abs)
print(len(genes), len(samples))


#
df = pd.concat([
    c_gdsc.loc[genes, samples].unstack().rename('crispr'),
    cnv_abs.loc[genes, samples].unstack().rename('cnv')
], axis=1).dropna().query('cnv != -1')
df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
df = df.assign(chr=sgrna_lib.groupby('GENES')['CHRM'].first()[df['gene']].values)

chr_cnv = df.groupby(['sample', 'chr'])['cnv'].median()
df = df.assign(chr_cnv=chr_cnv.loc[[(s, c) for s, c in df[['sample', 'chr']].values]].values)

#
plot_df = df.groupby(['sample', 'chr', 'cnv'])[['crispr', 'chr_cnv']].agg({'crispr': np.mean, 'chr_cnv': 'first'}).reset_index()
plot_df = plot_df[~plot_df['chr'].isin(['Y', 'X'])]
plot_df = plot_df.assign(ratio=plot_df['cnv'] / plot_df['chr_cnv'])

sns.jointplot('ratio', 'crispr', data=plot_df)
plt.show()
plt.savefig('reports/ratio_bias.png', bbox_inches='tight', dpi=600)
plt.close('all')

# -
sample = 'MDA-MB-415'

df = pd.concat([
    c_gdsc[sample].rename('crispr'),
    cnv_abs[sample].rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].first().rename('start')
], axis=1).dropna().sort_values(['chr', 'start'])
df = df.query('cnv != -1')
df = df[~df['chr'].isin(['Y', 'X'])]
df = df.assign(cnv_d=['%d' % i if i < 8 else '>=8' for i in df['cnv']])

thres = 4
plot_df = df[df['cnv'] > thres]
sns.boxplot('chr', 'crispr', data=plot_df, order=natsorted(set(plot_df['chr'])), color=bipal_dbgd[0])
sns.stripplot('chr', 'crispr', data=plot_df, order=natsorted(set(plot_df['chr'])), edgecolor='white', jitter=.4, color=bipal_dbgd[0])
plt.show()
# plt.xlabel('Chromosome')
# plt.ylabel('Crispy identified bias')
# plt.title('%s (copy-number > %d)' % (sample, thres))
# plt.savefig('reports/karyotype_bias.png', bbox_inches='tight', dpi=600)
# plt.close('all')


# -
plot_df = pd.concat([
    c_gdsc.loc[genes, samples].unstack().rename('bias'),
    cnv_abs.loc[genes, samples].unstack().rename('cnv')
], axis=1).query('cnv != -1').dropna()
plot_df = plot_df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
plot_df = plot_df.assign(cnv_d=plot_df['cnv'].apply(lambda x: '11+' if x > 10 else str(x)))
plot_df = plot_df.assign(rank=plot_df.groupby('sample')['bias'].rank(pct=True))

order = natsorted(set(plot_df['cnv_d']))

# Overall copy-number bias
ax = plot_cnv_rank(plot_df['cnv_d'], plot_df['bias'], notch=True, order=order)
ax.set_ylabel('Ranked copy-number bias')
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/cnv_boxplot_bias_gdsc.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Sample specific copy-number bias
df = pd.pivot_table(plot_df, index='cnv_d', columns='sample', values='bias')
df = df.loc[order, df.mean().sort_values().index]

sns.set(style='white', font_scale=.75)
ax = sns.heatmap(df, mask=df.isnull(), cmap='RdYlBu', lw=.3, center=0, square=False, cbar=True)
ax.set_xlabel('')
ax.set_ylabel('Copy-number')
plt.setp(ax.get_yticklabels(), rotation=0)
plt.gcf().set_size_inches(10, 1.5)
plt.savefig('reports/cnv_boxplot_bias_heatmap.png', bbox_inches='tight', dpi=600)
plt.close('all')
