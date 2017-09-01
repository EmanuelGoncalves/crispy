#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, zscore
from sklearn.metrics import roc_curve, auc
from limix_core.util.preprocess import gaussianize


# - Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# - CCLE and GDSC overlapping
cmap = pd.read_csv('data/ccle/CCLE_GDSC_cellLineMappoing.csv', index_col=2)

# - shRNA
# Project DRIVE
d_drive = pd.read_csv('data/ccle/DRIVE_ATARiS_data.txt', sep='\t')
d_drive.columns = [c.upper() for c in d_drive]

d_drive_rsa = pd.read_csv('data/ccle/DRIVE_RSA_data.txt', sep='\t')
d_drive_rsa.columns = [c.upper() for c in d_drive_rsa]
print('Project DRIVE: ', d_drive_rsa.shape)

# Project Achilles
d_achilles = pd.read_csv('data/ccle/Achilles_v2.20.2_GeneSolutions.txt', sep='\t', index_col=0).drop('Description', axis=1)
print('Project Achilles: ', d_achilles.shape)

# - CRISPR
# GDSC
crispr = pd.read_csv('data/gdsc/crispr/crispy_fold_change.csv', index_col=0)
crispr = crispr.groupby([i.split('_')[0].replace('sg', '') for i in crispr.index]).mean()

crispr_n = crispr.groupby([i.split('_')[0].replace('sg', '') for i in crispr.index]).mean()
crispr_n = pd.DataFrame(gaussianize(crispr_n.values.T).T, index=crispr_n.index, columns=crispr_n.columns)

# crispr = pd.DataFrame(gaussianize(crispr.values.T).T, index=crispr.index, columns=crispr.columns)

# crispr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)
# crispr = (crispr - crispr.unstack().mean()) / crispr.unstack().std()

# crispr = pd.read_csv('data/gdsc/crispr/deseq_gene_fold_changes_mean.csv', index_col=0)
# crispr = pd.DataFrame(gaussianize(crispr.values), index=crispr.index, columns=crispr.columns)
print('CRISPR: ', crispr.shape)

# Achilles
c_achilles = pd.read_csv('data/ccle/Achilles_v3.3.8_genesoln.gct', sep='\t', index_col=0).drop('Description', axis=1)

g_counts = pd.Series(dict(zip(*(np.unique([i.split('_')[0] for i in c_achilles.index], return_counts=True))))).sort_values()

c_achilles = c_achilles[[g_counts.loc[i.split('_')[0]] == 1 for i in c_achilles.index]]
c_achilles.index = [i.split('_')[0] for i in c_achilles.index]

# c_achilles = pd.DataFrame(gaussianize(c_achilles.values), index=c_achilles.index, columns=c_achilles.columns)

# c_achilles = pd.read_csv('data/ccle/Achilles_v3.3.8a.gct', sep='\t', index_col=0).drop('Description', axis=1)
# c_achilles = pd.DataFrame(zscore(c_achilles), index=c_achilles.index, columns=c_achilles.columns)
# c_achilles = c_achilles.groupby([i.split('_')[1] for i in c_achilles.index]).median()
# c_achilles = pd.DataFrame(zscore(c_achilles.T).T, index=c_achilles.index, columns=c_achilles.columns)
# c_achilles = pd.DataFrame(gaussianize(c_achilles.values), index=c_achilles.index, columns=c_achilles.columns)

# c_achilles = pd.read_csv('data/ccle/Achilles_v3.3.8.gct', sep='\t', index_col=0).drop('Description', axis=1)
# c_achilles = c_achilles.groupby([i.split('_')[1] for i in c_achilles.index]).median()


# - Gene essential Achilles AROC
samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_achilles)
genes = set(crispr.index).intersection(d_achilles.index)
print('Overlap: ', len(genes), len(samples))

plot_df = pd.DataFrame({
    'CRISPR': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
    'CRISPR_n': crispr_n.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
    'shRNA': d_achilles.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
}).dropna()
plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-6, 6).astype(int)
plot_df['essential'] = [int(i in essential) for _, i in plot_df.index]
print(plot_df.query('shRNA_int == 6').sort_values('CRISPR'))


# - Overlap
genes, samples = set(d_drive.index).intersection(d_achilles.index), set(d_drive).intersection(d_achilles)
print('Overlap: ', len(genes), len(samples))


# - Correlation
corr = []
for g in genes:
    df = pd.DataFrame({'drive': d_drive.loc[g, samples], 'achilles': d_achilles.loc[g, samples]}).dropna()

    cor, pval = pearsonr(df['drive'], df['achilles'])

    corr.append({'gene': g, 'cor': cor, 'pval': pval, 'len': df.shape[0]})

corr = pd.DataFrame(corr).sort_values('cor', ascending=False)
print(corr)

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
sns.distplot(corr['cor'], kde=False, bins=30, hist_kws={'color': '#34495e'})
plt.title('Mean Pearson R = %.2f' % corr['cor'].mean())
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/shrna_corr_hist.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Map overlaping cell lines
genes = set(d_drive.index).intersection(d_achilles.index).intersection(crispr.index)
samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_drive).intersection(d_achilles)
print('Overlap: ', len(genes), len(samples))

# - Correlation CRISPR with overlap DRIVE and Achilles
c_corr = []
for g in genes:
    df = pd.DataFrame({
        'drive': d_drive.loc[g, samples].rename(index=cmap.loc[samples, 'GDSC1000 name'].to_dict()),
        'achilles': d_achilles.loc[g, samples].rename(index=cmap.loc[samples, 'GDSC1000 name'].to_dict()),
        'crispr': crispr.loc[g, list(cmap.loc[samples, 'GDSC1000 name'])]
    }).dropna()

    d_cor, d_pval = pearsonr(df['crispr'], df['drive'])
    a_cor, a_pval = pearsonr(df['crispr'], df['achilles'])

    c_corr.append({
        'gene': g, 'len': df.shape[0],
        'cor_drive': d_cor, 'pval_drive': d_pval,
        'cor_achilles': a_cor, 'pval_achilles': a_pval,
    })

c_corr = pd.DataFrame(c_corr).sort_values('cor_drive', ascending=False)
print(c_corr)
print('Corr DRIVE: ', c_corr['cor_drive'].mean(), 'Achilles: ', c_corr['cor_achilles'].mean())

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
sns.distplot(c_corr['cor_drive'], kde=False, bins=50, color='#37454B', label='DRIVE')
sns.distplot(c_corr['cor_achilles'], kde=False, bins=50, color='#F2C500', label='Achilles')
plt.xlabel('Pearson between shRNA and CRISPR')
plt.title('Mean Pearson: DRIVE=%.2f, Achilles=%.2f' % (c_corr['cor_drive'].mean(), c_corr['cor_achilles'].mean()))
plt.legend()
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/shrna_crispr_corr_hist.png', bbox_inches='tight', dpi=600)
plt.close('all')



# - Gene essential Achilles AROC
samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_achilles)
genes = set(crispr.index).intersection(d_achilles.index)
print('Overlap: ', len(genes), len(samples))

plot_df = pd.DataFrame({
    'CRISPR': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
    'shRNA': d_achilles.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
}).dropna()
plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-6, 6).astype(int)

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
f, axs = plt.subplots(1, 2)

# AROCs
ax = axs[0]

thresholds = np.arange(-6, 1)
for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
    fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
    ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Classify shRNA genes with CRISPR')
ax.legend(loc=4, title='Achilles')

# Boxplots
ax = axs[1]

sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='RdGy', notch=True)
ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
ax.set_xlabel('Rounded and capped shRNA DEMETER')
ax.set_ylabel('CRISPR log2FC')
ax.set_title('CRISPR fold-change distributions')

plt.gcf().set_size_inches(7, 3)
plt.savefig('reports/shrna_crispr_achilles_agreement.png', bbox_inches='tight', dpi=600)
plt.close('all')


# # - Gene essential DRIVE AROC
# samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_drive_rsa)
# genes = set(crispr.index).intersection(d_drive_rsa.index)
# print('Overlap: ', len(genes), len(samples))
#
# plot_df = pd.DataFrame({
#     'CRISPR': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
#     'shRNA': d_drive_rsa.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
# }).dropna()
# plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
# plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-6, 6).astype(int)
#
# # Plot
# sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
# f, axs = plt.subplots(1, 2)
#
# # AROCs
# ax = axs[0]
#
# thresholds = np.arange(-6, 0)
# for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
#     fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
#     ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)
#
# ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_xlabel('False positive rate')
# ax.set_ylabel('True positive rate')
# ax.set_title('Classify shRNA genes with CRISPR')
# ax.legend(loc=4, title='DRIVE')
#
# # Boxplots
# ax = axs[1]
#
# sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='Reds_r', notch=True)
# ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
# ax.set_xlabel('Rounded and capped shRNA RSA')
# ax.set_ylabel('CRISPR log2FC')
# ax.set_title('CRISPR fold-change distributions')
#
# plt.gcf().set_size_inches(7, 3)
# plt.savefig('reports/shrna_crispr_drive_agreement.png', bbox_inches='tight', dpi=600)
# plt.close('all')
#
#
# # - Gene essential DRIVE AROC
# samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(d_drive)
# genes = set(crispr.index).intersection(d_drive.index)
# print('Overlap: ', len(genes), len(samples))
#
# plot_df = pd.DataFrame({
#     'CRISPR': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
#     'shRNA': d_drive.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
# }).dropna()
# plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
# plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-3, 3).astype(int)
#
# # Plot
# sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
# f, axs = plt.subplots(1, 2)
#
# # AROCs
# ax = axs[0]
#
# thresholds = np.arange(-3, 0)
# for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
#     fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
#     ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)
#
# ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_xlabel('False positive rate')
# ax.set_ylabel('True positive rate')
# ax.set_title('Classify shRNA genes with CRISPR')
# ax.legend(loc=4, title='DRIVE')
#
# # Boxplots
# ax = axs[1]
#
# sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='RdGy', notch=True)
# ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
# ax.set_xlabel('Rounded and capped shRNA ATARIS')
# ax.set_ylabel('CRISPR log2FC')
# ax.set_title('CRISPR fold-change distributions')
#
# plt.gcf().set_size_inches(7, 3)
# plt.savefig('reports/shrna_crispr_drive_ataris_agreement.png', bbox_inches='tight', dpi=600)
# plt.close('all')
#
#
# # - Gene essential Achilles CRISPR AROC
# samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(c_achilles)
# genes = set(crispr.index).intersection(c_achilles.index)
# print('Overlap: ', len(genes), len(samples))
#
# plot_df = pd.DataFrame({
#     'CRISPR': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
#     'shRNA': c_achilles.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
# }).dropna()
# plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
# plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-3, 3).astype(int)
#
# # Plot
# sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
# f, axs = plt.subplots(1, 2)
#
# # AROCs
# ax = axs[0]
#
# thresholds = np.arange(-3, 0)
# for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
#     fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
#     ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)
#
# ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_xlabel('False positive rate')
# ax.set_ylabel('True positive rate')
# ax.set_title('Classify CRISPR genes with CRISPR')
# ax.legend(loc=4, title='Achilles')
#
# # Boxplots
# ax = axs[1]
#
# sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='RdGy', notch=True)
# ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
# ax.set_xlabel('Rounded and capped sgRNA ATARIS')
# ax.set_ylabel('CRISPR log2FC')
# ax.set_title('CRISPR fold-change distributions')
#
# plt.gcf().set_size_inches(7, 3)
# plt.savefig('reports/shrna_crispr_achilles_crispr_agreement.png', bbox_inches='tight', dpi=600)
# plt.close('all')


# - Gene essential Achilles AROC
samples = set(c_achilles).intersection(d_achilles)
genes = set(c_achilles.index).intersection(d_achilles.index)
print('Overlap: ', len(genes), len(samples))

plot_df = pd.DataFrame({
    'CRISPR': c_achilles.loc[genes, samples].unstack(),
    'shRNA': d_achilles.loc[genes, samples].unstack()
}).dropna()
plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-6, 6).astype(int)

# Plot
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
f, axs = plt.subplots(1, 2)

# AROCs
ax = axs[0]

thresholds = np.arange(-6, 1)
for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
    fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
    ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Classify shRNA genes with CRISPR (all Achilles)')
ax.legend(loc=4, title='Achilles')

# Boxplots
ax = axs[1]

sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='RdGy', notch=True)
ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
ax.set_xlabel('Rounded and capped shRNA DEMETER')
ax.set_ylabel('CRISPR ATARIS distribution')
ax.set_title('CRISPR fold-change distributions')

plt.gcf().set_size_inches(7, 3)
plt.savefig('reports/shrna_all_achilles_agreement.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in crispr.columns]).intersection(c_achilles)
genes = set(crispr.index).intersection(c_achilles.index)
print('Overlap: ', len(genes), len(samples))

plot_df = pd.DataFrame({
    'CRISPR_GDSC': crispr.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
    'CRISPR_Achilles': c_achilles.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
}).dropna()

plot_df = pd.concat([
    plot_df[['CRISPR_GDSC']].assign(type='CRISPR_GDSC').rename(columns={'CRISPR_GDSC': 'value'}),
    plot_df[['CRISPR_Achilles']].assign(type='CRISPR_Achilles').rename(columns={'CRISPR_Achilles': 'value'})
])

plot_df['essential'] = [int(g in essential) for c, g in plot_df.index]

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
g = sns.boxplot('type', 'value', 'essential', data=plot_df, palette='RdGy', fliersize=1, linewidth=.3, notch=True)
plt.xlabel('')
plt.setp(g.get_xticklabels(), rotation=70)
plt.gcf().set_size_inches(1, 3)
plt.savefig('reports/essential_boxplots.png', bbox_inches='tight', dpi=600)
plt.close('all')
