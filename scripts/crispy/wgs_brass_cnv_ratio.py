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
# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_wgs.csv', index_col=0).apply(np.log2)

# Copy-number
cnv = pd.read_csv('data/crispy_copy_number_gene_wgs.csv', index_col=0)

# Ploidy
ploidy = pd.read_csv('data/crispy_copy_number_ploidy_wgs.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

# cnv_ratios = cnv.divide(ploidy).apply(np.log2)


# -
order = ['deletion', 'tandem-duplication', 'translocation', 'inversion']

# Import
brass_df = pd.read_csv('data/gdsc/wgs/wgs_brass_bedpe.csv')

# Parse sample
brass_df['sample'] = brass_df['sample'].apply(lambda v: v.split(',')[0])

# Filter
brass_df = brass_df[brass_df['svclass'].isin(order)]

brass_df = brass_df[brass_df['gene_id1'] == brass_df['gene_id2']]

brass_df = brass_df.query("gene_id1 != '_'")

brass_df = brass_df[brass_df['bkdist'] > 2500]

brass_df = brass_df.groupby(['sample', 'gene_id1'])['svclass'].agg(lambda x: set(x)).to_dict()

brass_df = {k: list(brass_df[k])[0] for k in brass_df if len(brass_df[k]) == 1}

# -
overlap = list({k[0] for k in brass_df}.intersection(cnv_ratios))


# -
sv_ratio = cnv_ratios[overlap].unstack().reset_index().dropna()
sv_ratio = sv_ratio.rename(columns={'level_0': 'sample', 'level_1': 'gene', 0: 'ratio'})
sv_ratio['svclass'] = [brass_df[(s, g)] if (s, g) in brass_df else '-' for s, g in sv_ratio[['sample', 'gene']].values]


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
sns.boxplot('ratio', 'svclass', data=sv_ratio, orient='h', color=bipal_dbgd[0], notch=True, linewidth=.5, fliersize=1)
plt.title('Structural rearrangements relation with\ncopy-number ratio')
plt.xlabel('Copy-number ratio (rank)')
plt.ylabel('')
plt.gcf().set_size_inches(3, 1)
plt.savefig('reports/crispy/brass_sv_ratio_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
