#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

# - Ploidy
ploidy = pd.read_csv('data/gdsc/ploidy.csv', index_col=0)[['PLOIDY']]

# - Import copy-number
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)

# - Import gene lists
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])
non_essential = list(pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'])

# - Import crispr
c_norm = pd.read_csv('data/gdsc/crispr/AMPhiBIAEcorrected_GLFC.csv', index_col=0)
c_lgfc = pd.read_csv('data/gdsc/crispr/deseq_gene_fold_changes_mean.csv', index_col=0)

# - Overlap
samples = set(cnv).intersection(c_norm).intersection(c_lgfc)
genes = set(cnv.index).intersection(c_norm.index).intersection(c_lgfc.index)

cnv, c_norm, c_lgfc = cnv.loc[genes, samples], c_norm.loc[genes, samples], c_lgfc.loc[genes, samples]
print(len(samples), len(genes))

# - Extract CNV info
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))
cnv_loh = cnv.applymap(lambda v: v.split(',')[2]).replace('0', 'H')

# - Rank data
c_norm_rank = pd.DataFrame({c: dict(zip(*(genes, rankdata(c_norm.loc[genes, c])))) for c in c_norm}) / len(genes)
c_lgfc_rank = pd.DataFrame({c: dict(zip(*(genes, rankdata(c_lgfc.loc[genes, c])))) for c in c_lgfc}) / len(genes)

# -
plot_df = pd.concat([
    cnv_abs.unstack(), c_norm.unstack(), c_lgfc.unstack(), c_norm_rank.unstack(), c_lgfc_rank.unstack(), cnv_loh.unstack()
], axis=1).rename(columns={0: 'cnv', 1: 'Fold-change corrected', 2: 'Fold-change', 3: 'Fold-change corrected (rank)', 4: 'Fold-change (rank)', 5: 'LOH'})

plot_df = plot_df[plot_df['cnv'] != -1]

pal = {'H': '#80b1d3', 'L': '#fb8072'}

sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
gs, pos = GridSpec(2, 2, hspace=.5, wspace=.3), 0

for f in ['Fold-change corrected', 'Fold-change', 'Fold-change corrected (rank)', 'Fold-change (rank)']:
    ax = plt.subplot(gs[pos])

    sns.boxplot('cnv', f, 'LOH', data=plot_df, linewidth=.3, notch=True, fliersize=1, orient='v', palette=pal)
    ax.legend_.remove()

    ax.set_ylabel('%s' % f)
    ax.set_xlabel('Copy-number (absolute)')

    pos += 1

handles = [mpatches.Circle([.0, .0], .25, facecolor=pal[s], label=s) for s in pal]
plt.legend(handles=handles, title='Type', loc='center left', bbox_to_anchor=(1.2, 0.5))

plt.gcf().set_size_inches(4, 2)
plt.savefig('reports/absolute_copy_number_crispr.png', bbox_inches='tight', dpi=600)
plt.close('all')
