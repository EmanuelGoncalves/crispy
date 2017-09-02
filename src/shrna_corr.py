#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats.stats import pearsonr, zscore
from limix_core.util.preprocess import gaussianize


# - CCLE and GDSC overlapping
cmap = pd.read_csv('data/ccle/CCLE_GDSC_cellLineMappoing.csv', index_col=2)

# - shRNA
# Project DRIVE
d_drive_rsa = pd.read_csv('data/ccle/DRIVE_RSA_data.txt', sep='\t')
d_drive_rsa.columns = [c.upper() for c in d_drive_rsa]
print('Project DRIVE: ', d_drive_rsa.shape)

# - CRISPR
# GDSC
# crispr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)
# crispr = pd.read_csv('data/gdsc/crispr/crispy_fold_change.csv', index_col=0)
# crispr = crispr.groupby([i.split('_')[0].replace('sg', '') for i in crispr.index]).mean()
crispr = pd.read_csv('data/gdsc/crispr/crispy_fold_change_glm_betas.csv', index_col=0)
print('CRISPR: ', crispr.shape)


# - Gene essential DRIVE AROC
samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_drive_rsa)
genes = set(crispr.index).intersection(d_drive_rsa.index)
print('Overlap: ', len(genes), len(samples))

plot_df = pd.DataFrame({
    'CRISPR': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
    'shRNA': d_drive_rsa.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
}).dropna()
plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-6, 6).astype(int)

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
f, axs = plt.subplots(1, 2)

# AROCs
ax = axs[0]

thresholds = np.arange(-6, 0)
for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
    fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
    ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Classify shRNA genes with CRISPR')
ax.legend(loc=4, title='DRIVE')

# Boxplots
ax = axs[1]

sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='Reds_r', notch=True)
ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
ax.set_xlabel('Rounded and capped shRNA RSA')
ax.set_ylabel('CRISPR log2FC')
ax.set_title('CRISPR fold-change distributions')

plt.gcf().set_size_inches(7, 3)
plt.savefig('reports/shrna_crispr_drive_agreement.png', bbox_inches='tight', dpi=600)
plt.close('all')
