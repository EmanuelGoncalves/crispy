#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import crispy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_gray
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
ccleanr.loc[genes, samples].subtract(ccrispy.loc[genes, samples]).median(1).sort_values()

g = 'HSPE1'
(ccleanr.loc[g, samples] - ccrispy.loc[g, samples]).sort_values()

s = 'CL-40'
ccleanr.loc[g, s] - ccrispy.loc[g, s]
fc.loc[g, s]


plot_df = pd.concat([
    fc[s].rename('fc'),
    ccrispy[s].rename('crispy'),
    ccrispy_kmean[s].rename('kmean'),
    ccleanr[s].rename('ccleanr'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos')
], axis=1).dropna()

#
g = sns.PairGrid(plot_df[['fc', 'crispy', 'ccleanr']], despine=False)
g = g.map_offdiag(plt.scatter, s=2, lw=.1, edgecolor='white', alpha=.4, color=bipal_gray[0])
g = g.map_diag(plt.hist, lw=.3, color=bipal_gray[0])
plt.gcf().set_size_inches(5, 5)
plt.savefig('reports/comparison_pairplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
chrm = plot_df.loc[g, 'chr']
plot_df_ = plot_df.query("chr == '%s'" % chrm)

plot_chromosome(plot_df_['pos'], plot_df_['fc'], plot_df_['kmean'], seg=cnv_seg.query("cellLine == '%s' & chr == '%s'" % (s, chrm)), highlight=[g])
plt.title('%s - chr: %s' % (s, chrm))
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/comparison_chromosome.png', bbox_inches='tight', dpi=600)
plt.close('all')
