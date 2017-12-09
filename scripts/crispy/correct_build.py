#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from natsort import natsorted
from sklearn.linear_model import LinearRegression
from crispy.benchmark_plot import plot_cumsum_auc
from sklearn.metrics import roc_curve, auc, roc_auc_score

# - Import
# Ploidy
ploidy = pd.read_csv('data/gdsc/cell_lines_project_ploidy.csv', index_col=0)['Average Ploidy']

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Non-essential genes
nessential = pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

# Copy-number absolute counts
gdsc_cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna()


# - Assemble
# GDSC
c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['regressed_out']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()
c_gdsc.round(5).to_csv('data/crispr_gdsc_crispy.csv')

# CCLE
c_ccle_fc = pd.read_csv('data/crispr_ccle_logfc.csv', index_col=0)

c_ccle = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_ccle_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['regressed_out']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_ccle_crispy_')
}).dropna()
c_ccle.round(5).to_csv('data/crispr_ccle_crispy.csv')


# - Copy-number absolute
gdsc_cnv_abs = gdsc_cnv[c_gdsc_fc.columns].applymap(lambda v: int(v.split(',')[1])).replace(-1, np.nan)


# - Benchmark
# Essential/Non-essential genes AUC
for n, df, df_ in [('gdsc', c_gdsc, c_gdsc_fc)]:
    # Essential AUCs corrected fold-changes
    ax, stats_ess = plot_cumsum_auc(df, essential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_essential_auc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Essential AUCs original fold-changes
    ax, stats_ess_fc = plot_cumsum_auc(df_, essential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_essential_auc_original_fc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Scatter plot of AUCs
    plot_df = pd.concat([pd.Series(stats_ess['auc'], name='Corrected'), pd.Series(stats_ess_fc['auc'], name='Original')], axis=1)

    ax = plt.gca()

    ax.scatter(plot_df['Original'], plot_df['Corrected'], c=bipal_dbgd[0], s=3, marker='x', lw=0.5)
    sns.rugplot(plot_df['Original'], height=.02, axis='x', c=bipal_dbgd[0], lw=.3, ax=ax)
    sns.rugplot(plot_df['Corrected'], height=.02, axis='y', c=bipal_dbgd[0], lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Original fold-changes AUCs')
    ax.set_ylabel('Corrected fold-changes AUCs')
    ax.set_title('Essential genes')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/processing_%s_crispy_essential_auc_scatter.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

for n, df, df_ in [('gdsc', c_gdsc, c_gdsc_fc)]:
    # Non-essential AUCs corrected fold-changes
    ax, stats_ness = plot_cumsum_auc(df, nessential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Non-essential AUCs original fold-changes
    ax, stats_ness_fc = plot_cumsum_auc(df_, nessential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc_original_fc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Scatter plot of AUCs
    plot_df = pd.concat([pd.Series(stats_ness['auc'], name='Corrected'), pd.Series(stats_ness_fc['auc'], name='Original')], axis=1)

    ax = plt.gca()

    ax.scatter(plot_df['Original'], plot_df['Corrected'], c=bipal_dbgd[0], s=3, marker='x', lw=0.5)
    sns.rugplot(plot_df['Original'], height=.02, axis='x', c=bipal_dbgd[0], lw=.3, ax=ax)
    sns.rugplot(plot_df['Corrected'], height=.02, axis='y', c=bipal_dbgd[0], lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Original fold-changes AUCs')
    ax.set_ylabel('Corrected fold-changes AUCs')
    ax.set_title('Non-essential genes')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc_scatter.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')


# Copy-number bias AUCs
plot_df = pd.DataFrame({c: c_gdsc_fc.loc[nexp[c], c] for c in c_gdsc_fc if c in nexp})
plot_df = pd.concat([
    plot_df.unstack().rename('fc'),
    gdsc_cnv.loc[plot_df.index, plot_df.columns].unstack().rename('cnv')
], axis=1).dropna()
plot_df = plot_df.assign(cnv_abs=plot_df['cnv'].apply(lambda v: int(v.split(',')[1])).values).query('cnv_abs >= 2')
plot_df = plot_df.assign(cnv=[str(i) if i < 10 else '10+' for i in plot_df['cnv_abs']])
plot_df = plot_df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
plot_df = plot_df.assign(ploidy=gdsc_cnv_abs.median()[plot_df['sample']].astype(int).values)

thresholds = natsorted(set(plot_df['cnv']))
thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(thresholds)).as_hex()

ax = plt.gca()

for c, thres in zip(thresholds_color, thresholds):
    fpr, tpr, _ = roc_curve((plot_df['cnv'] == thres).astype(int), -plot_df['fc'])
    ax.plot(fpr, tpr, label='%s: AUC=%.2f' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
legend = ax.legend(loc=4, title='Copy-number', prop={'size': 6})
legend.get_title().set_fontsize('7')

# plt.show()
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias.png', bbox_inches='tight', dpi=600)
plt.close('all')


# Copy-number variation across cell lines
df = {}
for c in set(plot_df['sample']):
    df_ = plot_df.query("sample == '%s'" % c)

    df[c] = {}

    for t in thresholds:
        if (sum(df_['cnv'] == t) >= 5) and (len(set(df_['cnv'])) > 1):
            df[c][t] = roc_auc_score((df_['cnv'] == t).astype(int), -df_['fc'])

df = pd.DataFrame(df)
df = df.unstack().reset_index().dropna().rename(columns={'level_0': 'sample', 'level_1': 'cnv', 0: 'auc'})
df = df.assign(ploidy=gdsc_cnv_abs.median()[df['sample']].astype(int).values)

sns.boxplot('cnv', 'auc', data=df, sym='', order=thresholds, palette=thresholds_color, linewidth=.3)
sns.stripplot('cnv', 'auc', data=df, order=thresholds, palette=thresholds_color, edgecolor='white', linewidth=.1, size=3, jitter=.4)
# plt.show()

plt.title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
plt.xlabel('Copy-number')
plt.ylabel('Copy-number AUC (cell lines)')
plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias_per_sample.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
pal = sns.light_palette(bipal_dbgd[0], n_colors=len(set(df['ploidy']))).as_hex()

sns.boxplot('cnv', 'auc', 'ploidy', data=df, sym='', order=thresholds, palette=pal, linewidth=.3)
sns.stripplot('cnv', 'auc', 'ploidy', data=df, order=thresholds, palette=pal, edgecolor='white', linewidth=.1, size=1, jitter=.2, split=True)
# plt.show()

handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(pal, thresholds))]
legend = plt.legend(handles=handles, title='Ploidy', prop={'size': 6}).get_title().set_fontsize('6')
plt.title('Cell ploidy effect CRISPR/Cas9 bias\nnon-expressed genes')
plt.xlabel('Copy-number')
plt.ylabel('Copy-number AUC (cell lines)')
plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias_per_sample_ploidy.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Copy-number variation relation with ploidy
ax = plt.gca()

sns.boxplot('cnv', 'fc', 'ploidy', data=plot_df, order=thresholds, palette=pal, linewidth=.3, fliersize=1, ax=ax, notch=True)
plt.axhline(0, lw=.3, c=bipal_dbgd[0], ls='-')
plt.xlabel('Copy-number')
plt.ylabel('CRISPR fold-change (log2)')
plt.title('Cell ploidy effect CRISPR/Cas9 bias\nnon-expressed genes')

plt.legend(title='Ploidy', prop={'size': 6}).get_title().set_fontsize('6')
plt.gcf().set_size_inches(4, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias_ploidy.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Ploidy effect in copy-number
plot_df = pd.concat([(gdsc_cnv_abs > 2).sum().rename('cnv'), ploidy.rename('ploidy')], axis=1).dropna()

g = sns.jointplot(
    'cnv', 'ploidy', data=plot_df, color=bipal_dbgd[0], space=0,
    joint_kws={'edgecolor': 'white', 'lw': .3, 's': 5},
    marginal_kws={'bins': 15}
)
g.set_axis_labels('# genes with copy-number > 2', 'Cell line ploidy')

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/ploidy_cnv_jointplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Ploidy agreement
df = pd.concat([
    ploidy.rename('ploidy'),
    gdsc_cnv.loc[:, ploidy.index].applymap(lambda v: int(v.split(',')[1]) if str(v).lower() != 'nan' else np.nan).replace(-1, np.nan).median().rename('cnv')
], axis=1)

g = sns.jointplot(
    'ploidy', 'cnv', data=df, color=bipal_dbgd[0], space=0,
    joint_kws={'edgecolor': 'white', 'lw': .3, 's': 5},
    marginal_kws={'bins': 15}
)
g.set_axis_labels('Ploidy COSMIC', 'Ploidy (median copy-number)')

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/ploidy_cosmic_corr.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Ploidy impact after correction
# Copy-number bias AUCs
plot_df = pd.DataFrame({c: c_gdsc.loc[nexp[c], c] for c in c_gdsc if c in nexp})
plot_df = pd.concat([
    plot_df.unstack().rename('crispy'),
    gdsc_cnv.loc[plot_df.index, plot_df.columns].unstack().rename('cnv')
], axis=1).dropna()
plot_df = plot_df.assign(cnv_abs=plot_df['cnv'].apply(lambda v: int(v.split(',')[1])).values).query('cnv_abs >= 2')
plot_df = plot_df.assign(cnv=[str(i) if i < 10 else '10+' for i in plot_df['cnv_abs']])
plot_df = plot_df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
plot_df = plot_df.assign(ploidy=gdsc_cnv_abs.median()[plot_df['sample']].astype(int).values)

# Enrichment
df_crispy = {}
for c in set(plot_df['sample']):
    df_ = plot_df.query("sample == '%s'" % c)

    df_crispy[c] = {}

    for t in thresholds:
        if (sum(df_['cnv'] == t) >= 5) and (len(set(df_['cnv'])) > 1):
            df_crispy[c][t] = roc_auc_score((df_['cnv'] == t).astype(int), -df_['crispy'])

df_crispy = pd.DataFrame(df_crispy)
df_crispy = df_crispy.unstack().reset_index().dropna().rename(columns={'level_0': 'sample', 'level_1': 'cnv', 0: 'auc'})
df_crispy = df_crispy.assign(ploidy=gdsc_cnv_abs.median()[df_crispy['sample']].astype(int).values)

sns.boxplot('cnv', 'auc', data=df_crispy, sym='', order=thresholds, palette=thresholds_color, linewidth=.3)
sns.stripplot('cnv', 'auc', data=df_crispy, order=thresholds, palette=thresholds_color, edgecolor='white', linewidth=.1, size=3, jitter=.4)
# plt.show()

plt.title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
plt.xlabel('Copy-number')
plt.ylabel('Copy-number AUC (cell lines)')
plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias_per_sample_corrected.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
pal = sns.light_palette(bipal_dbgd[0], n_colors=len(set(df_crispy['ploidy']))).as_hex()

sns.boxplot('cnv', 'auc', 'ploidy', data=df_crispy, sym='', order=thresholds, palette=pal, linewidth=.3)
sns.stripplot('cnv', 'auc', 'ploidy', data=df_crispy, order=thresholds, palette=pal, edgecolor='white', linewidth=.1, size=1, jitter=.2, split=True)
# plt.show()

handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(pal, thresholds))]
legend = plt.legend(handles=handles, loc=3, title='Ploidy', prop={'size': 6}).get_title().set_fontsize('6')
plt.title('Cell ploidy effect CRISPR/Cas9 bias\nnon-expressed genes')
plt.xlabel('Copy-number')
plt.ylabel('Copy-number AUC (cell lines)')
plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias_per_sample_ploidy_corrected.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
plot_df = pd.concat([
    df_crispy.set_index(['sample', 'cnv'])['auc'].rename('Corrected'),
    df.set_index(['sample', 'cnv'])['auc'].rename('Original')
], axis=1)

g = sns.jointplot(
    plot_df['Original'], plot_df['Corrected'], color=bipal_dbgd[0], space=0, kind='scatter',
    joint_kws={'s': 5, 'edgecolor': 'w', 'linewidth': .3}, stat_func=None
)

x_lim, y_lim = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
g.ax_joint.plot(x_lim, y_lim, ls='--', lw=.3, c=bipal_dbgd[0])
g.ax_joint.axhline(0.5, ls='-', lw=.1, c=bipal_dbgd[0])
g.ax_joint.axvline(0.5, ls='-', lw=.1, c=bipal_dbgd[0])

g.ax_joint.set_xlim(x_lim)
g.ax_joint.set_ylim(y_lim)

g.set_axis_labels('Original fold-changes AUCs', 'Corrected fold-changes AUCs')

plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/processing_crispy_nonessential_auc_scatter_corrected.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
ax = plt.gca()

ax.scatter(plot_df['Original'], plot_df['Corrected'], c=bipal_dbgd[0], s=3, marker='x', lw=0.5)
sns.rugplot(plot_df['Original'], height=.02, axis='x', c=bipal_dbgd[0], lw=.3, ax=ax)
sns.rugplot(plot_df['Corrected'], height=.02, axis='y', c=bipal_dbgd[0], lw=.3, ax=ax)

x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
ax.plot(x_lim, y_lim, 'k--', lw=.3)

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)

ax.set_xlabel('Original fold-changes AUCs')
ax.set_ylabel('Corrected fold-changes AUCs')
ax.set_title('Copy-number bias\nnon-essential genes')

plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/processing_crispy_nonessential_auc_scatter_corrected.png', bbox_inches='tight', dpi=600)
plt.close('all')
