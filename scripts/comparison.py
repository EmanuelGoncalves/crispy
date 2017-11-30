#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_gray
from matplotlib.gridspec import GridSpec
from crispy.benchmark_plot import plot_cumsum_auc, plot_cnv_rank, plot_chromosome


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# CRISPR fold-changes
fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# CRISPRcleanR
ccleanr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)

# Crispy
ccrispy = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)
ccrispy_kmean = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna()

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)


# - Overlap
samples, genes = list(set(ccleanr).intersection(ccrispy).intersection(cnv)), list(set(ccleanr.index).intersection(ccrispy.index).intersection(cnv.index))
print(len(samples), len(genes))

# Copy-number
cnv_abs = cnv[samples].applymap(lambda v: int(v.split(',')[0]))


# -
aff_genes = ccleanr.loc[genes, samples].subtract(ccrispy.loc[genes, samples]).mean(1).sort_values(ascending=False)

g = 'GAPDH'

ccleanr.loc[genes, samples].subtract(ccrispy.loc[genes, samples]).loc[g].sort_values()

s = 'OVCAR-8'

plot_df = pd.concat([
    fc[s].rename('fc'),
    ccrispy[s].rename('crispy'),
    ccrispy_kmean[s].rename('kmean'),
    ccleanr[s].rename('ccleanr'),
    cnv_abs[s].rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos')
], axis=1).dropna().query('cnv != -1').dropna()

#
gs = sns.PairGrid(plot_df[['fc', 'crispy', 'ccleanr']], despine=False)

gs = gs.map_offdiag(plt.scatter, s=2, lw=.1, edgecolor='white', alpha=.4, color=bipal_gray[0])
gs = gs.map_diag(plt.hist, lw=.3, color=bipal_gray[0])

gs.data = plot_df.loc[aff_genes.head(20).index, ['fc', 'crispy', 'ccleanr']]
gs = gs.map_offdiag(plt.scatter, s=3, lw=.1, edgecolor='white', alpha=.8, color=bipal_gray[1])

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/comparison_pairplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Copy-number boxplot effects
gs = GridSpec(3, 1, hspace=.3, wspace=.3)
for i, l in enumerate(['fc', 'crispy', 'ccleanr']):
    ax = plt.subplot(gs[i])
    ax = plot_cnv_rank(plot_df['cnv'].astype(int), plot_df[l], color=bipal_gray[1], ax=ax, notch=True)
    ax.set_ylabel(l)
plt.gcf().set_size_inches(3, 4)
plt.savefig('reports/comparison_cnv.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
chrm = plot_df.loc[g, 'chr']
plot_df_ = plot_df.query("chr == '%s'" % chrm)

ax = plot_chromosome(plot_df_['pos'], plot_df_['fc'], plot_df_['kmean'], seg=cnv_seg.query("cellLine == '%s' & chr == '%s'" % (s, chrm)))

ax.scatter(plot_df_.loc[g, 'pos'], plot_df_.loc[g, 'fc'], s=5, marker='X', lw=0, c='#8da0cb', alpha=.5, label='Orignal')
ax.scatter(plot_df_.loc[g, 'pos'], plot_df_.loc[g, 'crispy'], s=5, marker='X', lw=0, c='#fc8d62', alpha=.5, label='Crispy')
ax.scatter(plot_df_.loc[g, 'pos'], plot_df_.loc[g, 'ccleanr'], s=5, marker='X', lw=0, c='#ffd92f', alpha=.5, label='CCleanR')

ax.set_ylabel('')

plt.title('%s - chr: %s' % (s, chrm))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.gcf().set_size_inches(3, 1)
plt.savefig('reports/comparison_chromosome.png', bbox_inches='tight', dpi=600)
plt.close('all')
