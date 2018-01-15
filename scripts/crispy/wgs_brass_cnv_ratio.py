#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from matplotlib.colors import rgb2hex
from sklearn.metrics import roc_curve, auc


# - Import
# Brass
brass_files = filter(lambda x: x.endswith('.brass.annot.bedpe'), os.listdir('data/gdsc/wgs/brass_bedpe/'))
brass_df = pd.concat([pd.read_csv('data/gdsc/wgs/brass_bedpe/%s' % f, skiprows=110, sep='\t') for f in brass_files]).reset_index(drop=True)
brass_df.to_csv('data/gdsc/wgs/wgs_brass_bedpe.csv')

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_ratio.csv', index_col=0).replace(-1, np.nan)

# Copy-number absolute counts
cnv_abs = pd.read_csv('data/crispy_gene_copy_number.csv', index_col=0).replace(-1, np.nan)

# GDSC CRISPR
c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# GDSC CRISPR mean bias
c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()


# - Overlap
samples = set(brass_df['sample']).intersection(cnv_ratios)
genes = set(cnv_ratios.index).intersection(brass_df['gene1']).intersection(brass_df['gene1'])
print('Samples: ', len(samples), 'Genes: ', len(genes))


# -
sv_ratio = brass_df.loc[list(map(lambda x: len(x.split(',')) == 1, brass_df['sample']))]

sv_ratio = pd.melt(sv_ratio, id_vars=['sample', 'svclass'], value_vars=['gene1', 'gene2']).query("value != '_'")
sv_ratio = sv_ratio[sv_ratio['value'].isin(genes) & sv_ratio['sample'].isin(samples)]

sv_ratio = sv_ratio.assign(ratio=[cnv_abs.loc[g, s] for g, s in sv_ratio[['value', 'sample']].values]).dropna().sort_values('ratio')
sv_ratio = sv_ratio.drop_duplicates(subset=['sample', 'svclass', 'value'])

# sv_ratio = sv_ratio.assign(crispr=c_gdsc_fc.unstack().loc[[(s, g) for g, s in sv_ratio[['value', 'sample']].values]].values)
# sv_ratio = sv_ratio.assign(crispy=c_gdsc.unstack().loc[[(s, g) for g, s in sv_ratio[['value', 'sample']].values]].values)


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
