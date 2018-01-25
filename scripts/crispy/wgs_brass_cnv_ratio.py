#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from matplotlib.colors import rgb2hex
from sklearn.metrics import roc_curve, auc
from pyensembl import EnsemblRelease


# - Import
# Brass
brass_df = pd.read_csv('data/gdsc/wgs/wgs_brass_bedpe.csv')

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_snp.csv', index_col=0)

# Pyensembl release 77: human reference genome GRCh37
densembl = EnsemblRelease(75)


# -
sv_ratio = []
for idx, row in brass_df.iterrows():
    sample = row['sample'].split(',')[0]

    if sample in cnv_ratios.columns:
        genes1 = densembl.gene_names_at_locus(contig=row['chr1'], position=int(row['start1']), end=int(row['end1']))
        genes2 = densembl.gene_names_at_locus(contig=row['chr2'], position=int(row['start2']), end=int(row['end2']))

        for g in set(genes1).union(genes2).intersection(cnv_ratios.index):
            sv_ratio.append({
                'sample': sample, 'gene': g,
                'svclass': row['svclass'], 'bkdist': row['bkdist'], 'assembly_score': row['assembly_score'],
                'ratio': cnv_ratios.loc[g, sample],
            })

sv_ratio = pd.DataFrame(sv_ratio).dropna()


# -
# plot_df = sv_ratio.query('bkdist > 2500')

order = ['deletion', 'inversion', 'translocation', 'tandem-duplication']

#
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
