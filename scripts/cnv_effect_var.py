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
from scipy.stats import mode
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

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)
cnv_seg = cnv_seg[cnv_seg['cellLine'].isin(set(c_gdsc))]

seg = cnv_seg[(cnv_seg['cellLine'] == '697') & (cnv_seg['chr'] == 1)]

ax = plt.gca()
for i, (s, e, c) in enumerate(seg[['startpos', 'endpos', 'totalCN']].values):
    ax.plot([s, e], [c, c], lw=1., c='#58595B', alpha=.9, label='totalCN' if i == 0 else None)
plt.show()

cnv_seg_c = []
for s in set(cnv_seg['cellLine']):
    for c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']:
        seg = cnv_seg[(cnv_seg['cellLine'] == s) & (cnv_seg['chr'] == c)]
        seg_c = [((e - s) * (c + 1), (e - s)) for i, (s, e, c) in enumerate(seg[['startpos', 'endpos', 'totalCN']].values)]
        d, n = zip(*(seg_c))
        cpv = sum(d) / sum(n) - 1
        cnv_seg_c.append({'sample': s, 'chr': c, 'ploidy': cpv})
cnv_seg_c = pd.DataFrame(cnv_seg_c)
cnv_seg_c = cnv_seg_c.groupby(['sample', 'chr']).first()


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

# chr_cnv = df.groupby(['sample', 'chr'])['cnv'].median()
# df = df.assign(chr_cnv=cnv_seg_c.loc[[(s, c) for s, c in df[['sample', 'chr']].values]].values)
chr_cnv = df.groupby(['sample', 'chr'])['cnv'].median()
df = df.assign(chr_cnv=chr_cnv.loc[[(s, c) for s, c in df[['sample', 'chr']].values]].values)

df = df.assign(ploidy=ploidy[df['sample']].values)

#
plot_df = df.groupby(['sample', 'chr', 'cnv'])[['crispr', 'chr_cnv', 'ploidy']].agg({'crispr': np.mean, 'chr_cnv': 'first', 'ploidy': 'first'}).reset_index()
plot_df = plot_df[~plot_df['chr'].isin(['Y', 'X'])]
plot_df = plot_df.assign(ratio=plot_df['cnv'] / plot_df['chr_cnv'])
plot_df = plot_df.dropna()

g = sns.jointplot('ratio', 'crispr', data=plot_df, kind='reg', color=bipal_dbgd[0], joint_kws={'line_kws': {'color': bipal_dbgd[1]}})
sns.kdeplot(plot_df['ratio'], plot_df['crispr'], ax=g.ax_joint, shade_lowest=False, n_levels=60)
plt.show()

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/ratio_bias.png', bbox_inches='tight', dpi=600)
plt.close('all')

plot_df_ = pd.concat([
    plot_df[(plot_df['ratio'] < 1) & (plot_df['cnv'] >= 3)].assign(type='depletion (p<1)'),
    plot_df[(plot_df['ratio'] < 3) & (plot_df['ratio'] >= 1) & (plot_df['cnv'] >= 3)].assign(type='duplication (p<3)'),
    plot_df[(plot_df['ratio'] >= 3) & (plot_df['cnv'] >= 3)].assign(type='elongation (p>3)')
])

sns.boxplot('crispr', 'type', data=plot_df_, orient='h', notch=True, color=bipal_dbgd[0], fliersize=1)
plt.show()

plt.xlabel('Crispr mean bias')
plt.ylabel('Chromosome')
plt.gcf().set_size_inches(3, 1)
plt.savefig('reports/ratio_bias_boxplot.png', bbox_inches='tight', dpi=600)
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
