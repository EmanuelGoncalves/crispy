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
# Brass
brass_df = pd.read_csv('data/gdsc/wgs/wgs_brass_bedpe.csv')

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_ratio.csv', index_col=0).replace(-1, np.nan)


# - Overlap
samples = set(brass_df['sample']).intersection(cnv_ratios)
genes = set(cnv_ratios.index).intersection(brass_df['gene1']).intersection(brass_df['gene1'])
print('Samples: ', len(samples), 'Genes: ', len(genes))


# -
sv_ratio = brass_df.loc[list(map(lambda x: len(x.split(',')) == 1, brass_df['sample']))]
sv_ratio = sv_ratio.query("copynumber_flag == 1 & assembly_score != '_'")
sv_ratio = sv_ratio[sv_ratio['gene1'] == sv_ratio['gene2']]

sv_ratio = pd.melt(sv_ratio, id_vars=['sample', 'svclass'], value_vars=['gene1']).query("value != '_'")
sv_ratio = sv_ratio[sv_ratio['value'].isin(genes) & sv_ratio['sample'].isin(samples)]
sv_ratio = sv_ratio.drop_duplicates(subset=['sample', 'svclass', 'value'])

sv_ratio = sv_ratio.assign(ratio=[cnv_ratios.loc[g, s] for g, s in sv_ratio[['value', 'sample']].values]).dropna().sort_values('ratio')
sv_ratio = sv_ratio.assign(ratio_rank=sv_ratio['ratio'].rank(pct=True))


# -
order = ['deletion', 'inversion', 'translocation', 'tandem-duplication']
ax = plt.gca()
for t, c in zip(order, map(rgb2hex, sns.light_palette(bipal_dbgd[0], len(order) + 1)[1:])):
    fpr, tpr, _ = roc_curve((sv_ratio['svclass'] == t).astype(int), sv_ratio['ratio'])
    ax.plot(fpr, tpr, label='%s=%.2f (AUC)' % (t, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Structural rearrangements relation with\ncopy-number ratio')

legend = ax.legend(loc=4, prop={'size': 6})

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/brass_sv_ratio_cumdist.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
sns.boxplot('ratio_rank', 'svclass', data=sv_ratio, orient='h', color=bipal_dbgd[0], notch=True, linewidth=.5, fliersize=1)
plt.title('Structural rearrangements relation with\ncopy-number ratio')
plt.xlabel('Copy-number ratio (rank)')
plt.ylabel('')
plt.gcf().set_size_inches(3, 1)
plt.savefig('reports/brass_sv_ratio_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
