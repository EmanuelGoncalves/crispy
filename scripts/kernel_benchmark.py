#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_gray
from matplotlib.gridspec import GridSpec
from crispy.biases_correction import CRISPRCorrection
from crispy.benchmark_plot import plot_cumsum_auc, plot_cnv_rank, plot_chromosome
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic, RBF, Matern


# - Imports
sample = 'AU565'

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))[sample]

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)[sample]

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)[sample]
cnv_abs = cnv.apply(lambda v: int(v.split(',')[0]))

# CRISPRcleanR
ccleanr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)[sample]

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg = cnv_seg.query("cellLine == '%s'" % sample)
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)


# - Biases correction method
# Build data-frame
df = pd.concat([
    fc_sgrna.groupby(sgrna_lib.loc[fc_sgrna.index, 'GENES']).mean().rename('logfc'),
    cnv_abs.rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos'),
], axis=1).dropna().query('cnv != -1')

# Correction
crispy_rquad = CRISPRCorrection(
    kernel=ConstantKernel() * RationalQuadratic(length_scale_bounds=(1e-5, 10), alpha_bounds=(1e-5, 10)) + WhiteKernel()
).rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['logfc'])
crispy_rquad_df = pd.concat([v.to_dataframe() for k, v in crispy_rquad.items()])
crispy_rquad_df.to_csv('data/kernel_benchmark_rquad.csv')

crispy_rbf = CRISPRCorrection(
    kernel=ConstantKernel() * RBF(length_scale_bounds=(1e-5, 10)) + WhiteKernel()
).rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['logfc'])
crispy_rbf_df = pd.concat([v.to_dataframe() for k, v in crispy_rbf.items()])
crispy_rbf_df.to_csv('data/kernel_benchmark_rbf.csv')

crispy_matern = CRISPRCorrection(
    kernel=ConstantKernel() * Matern(length_scale_bounds=(1e-5, 100), nu=.5) + WhiteKernel()
).rename(sample).fit_by(by=df['chr'], X=df[['pos']] / 1e5, y=df['logfc'])
crispy_matern_df = pd.concat([v.to_dataframe() for k, v in crispy_matern.items()])
crispy_matern_df.to_csv('data/kernel_benchmark_matern.csv')


# - Plots
plot_df = pd.concat([
    df,
    (crispy_rquad_df['logfc'] - crispy_rquad_df['k_mean']).rename('RQuad'),
    (crispy_rbf_df['logfc'] - crispy_rbf_df['k_mean']).rename('RBF'),
    (crispy_matern_df['logfc'] - crispy_matern_df['k_mean']).rename('Matern'),

    (crispy_rquad_df['k_mean']).rename('RQuad_kmean'),
    (crispy_rbf_df['k_mean']).rename('RBF_kmean'),
    (crispy_matern_df['k_mean']).rename('Matern_kmean'),

    ccleanr.rename('CRISPRcleanR'),
], axis=1).dropna()

# Correlation
g = sns.PairGrid(plot_df[['logfc', 'RQuad', 'RBF', 'Matern', 'CRISPRcleanR']], despine=False)
g = g.map_offdiag(plt.scatter, s=2, lw=.1, edgecolor='white', alpha=.4, color=bipal_gray[0])
g = g.map_diag(plt.hist, lw=.3, color=bipal_gray[0])
plt.gcf().set_size_inches(5, 5)
plt.savefig('reports/kernel_benchmark_correlation.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Essential genes AUC
plot_cumsum_auc(plot_df[['logfc', 'RQuad', 'RBF', 'Matern', 'CRISPRcleanR']], essential)
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/kernel_benchmark_essential_auc.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Copy-number boxplot effects
gs = GridSpec(5, 1, hspace=.3, wspace=.3)
for i, l in enumerate(['logfc', 'RQuad', 'RBF', 'Matern', 'CRISPRcleanR']):
    ax = plt.subplot(gs[i])
    plot_cnv_rank(plot_df['cnv'].astype(int), plot_df[l], ax=ax, color=bipal_gray[1])
plt.gcf().set_size_inches(4, 10)
plt.savefig('reports/kernel_benchmark_cnv.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Plot chromossome
gs = GridSpec(3, 1, hspace=.3, wspace=.3)
for i, l in enumerate(['RQuad_kmean', 'RBF_kmean', 'Matern_kmean']):
    ax = plt.subplot(gs[i])
    plot_df_ = plot_df.query("chr == '8'").sort_values('pos')
    ax = plot_chromosome(plot_df_['pos'], plot_df_['logfc'], plot_df_[l], seg=cnv_seg.query("chr == '8'"), highlight=['MYC'], ax=ax)
    ax.set_ylabel('%s (log2 FC)' % l)

plt.gcf().set_size_inches(4, 6)
plt.savefig('reports/kernel_benchmark_chromossome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')
