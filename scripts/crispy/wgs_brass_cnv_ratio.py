#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from matplotlib.colors import rgb2hex
from sklearn.metrics import roc_curve, auc


# - Import
# Ploidy
ploidy = pd.read_csv('data/crispy_copy_number_ploidy_wgs.csv', index_col=0, names=['sample', 'ploidy'])['ploidy'].apply(np.log2)

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_wgs.csv', index_col=0).apply(np.log10)

# Copy-number
cnv = pd.read_csv('data/crispy_copy_number_gene_wgs.csv', index_col=0).apply(np.log10)

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_rpkm_preprocessed.csv', index_col=0).dropna()

# Brass
brass = pd.read_csv('data/gdsc/wgs/wgs_brass_bedpe.csv')
brass['sample'] = brass['sample'].apply(lambda v: v.split(',')[0])


# -
genes = set(cnv.index).intersection(cnv_ratios.index).intersection(gexp.index)
samples = list(set(cnv_ratios).intersection(brass['sample']).intersection(gexp))
print('Samples: {}; Genes: {}'.format(len(samples), len(genes)))


# -
order = ['deletion', 'tandem-duplication', 'translocation', 'inversion']

# Filter
brass_df = brass[brass['svclass'].isin(order)]
brass_df = brass_df.query("assembly_score != '_'")
brass_df = brass_df.query("bkdist != -1")
brass_df = brass_df[brass_df['gene_id1'] == brass_df['gene_id2']]
brass_df = brass_df.groupby(['sample', 'gene_id1'])['svclass'].agg(lambda x: set(x)).to_dict()
brass_df = {k: list(brass_df[k])[0] for k in brass_df if len(brass_df[k]) == 1}


# -
sv_ratio = cnv_ratios[samples].unstack().reset_index().dropna()
sv_ratio = sv_ratio.rename(columns={'level_0': 'sample', 'level_1': 'gene', 0: 'ratio'})
sv_ratio['svclass'] = [brass_df[(s, g)] if (s, g) in brass_df else '-' for s, g in sv_ratio[['sample', 'gene']].values]


# - Corr
corr_cnv = gexp.loc[genes, samples].T.corrwith(cnv.loc[genes, samples].T)
corr_ratios = gexp.loc[genes, samples].T.corrwith(cnv_ratios.loc[genes, samples].T)

# Plot
plot_df = pd.concat([corr_cnv.rename('cnv'), corr_ratios.rename('ratios')], axis=1).dropna()

g = sns.jointplot(
    'cnv', 'ratios', data=plot_df, kind='reg', space=0, color=bipal_dbgd[0], annot_kws=dict(stat='r'),
    marginal_kws={'kde': False}, joint_kws={'scatter_kws': {'edgecolor': 'w', 'lw': .3, 's': 8, 'alpha': .3}, 'line_kws': {'lw': 1.}}
)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/crispy/brass_sv_ratio_corr_jointplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
sns.distplot(plot_df.eval('cnv - ratios'), bins=30, color=bipal_dbgd[0])
plt.axvline(0, lw=.1, ls='--', color=bipal_dbgd[0], alpha=.8)
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/crispy/brass_sv_ratio_corr_histogram.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
plot_df = sv_ratio[sv_ratio['ratio'].apply(np.isfinite)]

ax = plt.gca()
for t, c in zip(order, map(rgb2hex, sns.light_palette(bipal_dbgd[0], len(order) + 1)[1:])):
    fpr, tpr, _ = roc_curve((plot_df['svclass'] == t).astype(int), plot_df['ratio'])
    ax.plot(fpr, tpr, label='%s=%.2f (AUC)' % (t, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Structural rearrangements relation with\ncopy-number ratio')

legend = ax.legend(loc=4, prop={'size': 6})

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/crispy/brass_sv_ratio_cumdist.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
sns.boxplot('ratio', 'svclass', data=sv_ratio, orient='h', order=order, color=bipal_dbgd[0], notch=True, linewidth=.5, fliersize=1)
plt.title('Structural rearrangements relation with\ncopy-number ratio')
plt.xlabel('Copy-number ratio (rank)')
plt.ylabel('')
plt.gcf().set_size_inches(3, 1)
plt.savefig('reports/crispy/brass_sv_ratio_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
