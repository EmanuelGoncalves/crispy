#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import re
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from limix_core.util.preprocess import gaussianize
from datetime import datetime as dt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from scipy_sugar.stats import quantile_gaussianize
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


# - Import gene lists
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])
non_essential = list(pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'])


# - Read cell lines manifest
manifest = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t')
manifest['CGaP_Sample_Name'] = [i.split('_')[0] for i in manifest['CGaP_Sample_Name']]
manifest = manifest.groupby('CGaP_Sample_Name')['Cell_Line_Name'].first()


# - Import other crispr scores
c_mageck = pd.read_csv('data/gdsc/crispr/mageck_gene_fold_changes_mean.csv', index_col=0)
c_bagel = -pd.read_csv('data/gdsc/crispr/crispr_bagel_gene_level_normalised.csv', index_col=0)


# - Import GDSC GISTIC copy-number
gdsc_cnv = pd.read_csv('./data/gdsc/copynumber/gistic/1509967/segmentation_file.all_thresholded.by_genes.txt', sep='\t', index_col=0).drop(['Locus ID', 'Cytoband'], axis=1).dropna()


# - Match with RACS
racs = pd.read_csv('data/gdsc/copynumber/cnv_bems_racs.csv', sep=',')


# - Load DESeq2 calculated fold-changes
c_deseq = {}
# f = 'CO320H_logfold.tsv'
for f in os.listdir('data/gdsc/crispr/deseq2/'):
    if f.endswith('_logfold.tsv'):
        # Get cell line name
        c = manifest[f.split('_')[0]]
        print('[%s] Loading %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c))

        # Load fold-changes
        df = pd.read_csv('data/gdsc/crispr/deseq2/%s' % f, sep='\t')['log2FoldChange'].dropna()

        # Consider Library v1.1 sgRNAs
        df.index = [i[2:] if re.match('sg', i) else i for i in df.index]

        # Store values
        c_deseq[c] = df

c_deseq = pd.DataFrame(c_deseq)
c_deseq.to_csv('data/gdsc/crispr/deseq_gene_fold_changes.csv')

# - Average sgRNAs per gene
c_deseq = c_deseq.groupby([i.split('_')[0] for i in c_deseq.index]).mean().dropna()
c_deseq.to_csv('data/gdsc/crispr/deseq_gene_fold_changes_mean.csv')
# c_deseq = pd.read_csv('data/gdsc/crispr/deseq_gene_fold_changes_mean.csv', index_col=0)


# - Fold-changes comparison
# Overlap
samples = set(c_deseq).intersection(c_bagel)
genes = set(c_deseq.index).intersection(c_bagel.index)

# Plot
plot_df = pd.concat([c_deseq.loc[genes, samples].unstack(), c_bagel.loc[genes, samples].unstack()], axis=1)
plot_df = plot_df.reset_index().rename(columns={0: 'deseq2', 1: 'bagel', 'level_0': 'sample', 'level_1': 'gene'})

sns.set(rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5}, style='ticks', context='paper', font_scale=.75)
g = sns.JointGrid('deseq2', 'bagel', plot_df, space=0, ratio=8)

g = g.plot_marginals(sns.kdeplot, shade=True, alpha=.5, lw=.1, color='#34495e')

g = g.plot_joint(sns.regplot, color='#34495e', truncate=True, fit_reg=False, scatter_kws={'s': 2, 'edgecolor': 'w', 'linewidth': .2, 'alpha': .5}, line_kws={'linewidth': .5})
g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)

g.set_axis_labels('DESeq2 fold-changes', 'Bagel scores')

plt.suptitle('FCs vs Bagel', y=1.05)

plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/deseq2_fold_changes_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Normalise
c_deseq = pd.DataFrame({c: quantile_gaussianize(c_deseq[c].values) for c in c_deseq}, index=c_deseq.index)


# - Boxplots of essential/non-essential genes
cmap = pd.Series.from_csv('data/gdsc/colour_palette_cell_lines.csv')

order = list(cmap[list(c_deseq)].sort_values().index)
hue_order = ['Other', 'Non-essential', 'Essential']

pal = {'Other': '#dfdfdf', 'Non-essential': '#34495e', 'Essential': '#e74c3c'}

plot_df = c_deseq.unstack().reset_index()
plot_df.columns = ['sample', 'gene', 'score']
plot_df['type'] = ['Essential' if g in essential else ('Non-essential' if g in non_essential else 'Other') for g in plot_df['gene']]

sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.boxplot('sample', 'score', 'type', data=plot_df, linewidth=.3, notch=True, fliersize=1, orient='v', order=order, hue_order=hue_order, palette=pal)
plt.axhline(0, lw=.3, ls='-', color='#dfdfdf')
# plt.xticks([])
plt.setp(g.get_xticklabels(), rotation=90)
plt.ylabel('fold-changes')
plt.xlabel('Samples')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(12, 2)
plt.savefig('reports/deseq2_normalised_boxplots.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Evaluate essential genes
# Overlap
samples = set(c_deseq).intersection(c_mageck).intersection(c_bagel)
genes = set(c_deseq.index).intersection(c_mageck.index).intersection(c_bagel.index)

# Binary essential status (speed-up)
status = [int(i in essential) for i in genes]

# AROC and AUPR
plot_df = pd.DataFrame([{
    'type': d,
    'sample': c,
    'aroc': roc_auc_score(status, -df.loc[genes, c].values),
    'aupr': average_precision_score(status, -df.loc[genes, c].values)
} for d, df in [('deseq2', c_deseq), ('mageck', c_mageck), ('bagel', c_bagel)] for c in samples])

# Plot
gs, pos = GridSpec(2, 2, hspace=.6, wspace=.6), 0
sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})

for x, y, t in [(i, j, t) for t in ['aroc', 'aupr'] for i, j in [('deseq2', 'mageck'), ('deseq2', 'bagel')]]:
    ax = plt.subplot(gs[pos])

    xs = plot_df[(plot_df['type'] == x)].set_index('sample').loc[samples, t]
    ys = plot_df[(plot_df['type'] == y)].set_index('sample').loc[samples, t]

    ax.scatter(xs, ys, c='#34495e', s=3, marker='x', lw=0.5)

    sns.rugplot(xs, height=.02, axis='x', c='#34495e', lw=.3, ax=ax)
    sns.rugplot(ys, height=.02, axis='y', c='#34495e', lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(t.upper())

    pos += 1

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/comparison_essential_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Boxplots
sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.boxplot('type', 'aroc', data=plot_df, linewidth=.3, notch=True, fliersize=1, orient='v', color='#34495e')
plt.axhline(0.5, lw=.3, ls='-', color='#dfdfdf')
# plt.xticks([])
plt.setp(g.get_xticklabels(), rotation=90)
plt.ylabel('AROC')
plt.xlabel('')
plt.title('Essential genes')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='GISTIC')
plt.gcf().set_size_inches(.75, 2)
plt.savefig('reports/comparison_essential_scatter_aroc.png', bbox_inches='tight', dpi=600)
plt.close('all')

sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.boxplot('type', 'aupr', data=plot_df, linewidth=.3, notch=True, fliersize=1, orient='v', color='#34495e')
plt.axhline(0.0, lw=.3, ls='-', color='#dfdfdf')
# plt.xticks([])
plt.setp(g.get_xticklabels(), rotation=90)
plt.ylabel('AUPR')
plt.xlabel('')
plt.title('Essential genes')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='GISTIC')
plt.gcf().set_size_inches(.75, 2)
plt.savefig('reports/comparison_essential_scatter_aupr.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Evaluate amplifications
cmap = {-2: '#e74c3c', -1: '#F0938A', 0: '#d5d5d5', 1: '#7bc399', 2: '#239c56'}

# Overlap
samples = set(c_deseq).intersection(c_mageck).intersection(c_bagel).intersection(gdsc_cnv)
genes = set(c_deseq.index).intersection(c_mageck.index).intersection(c_bagel.index).intersection(gdsc_cnv.index)


# AROC and AUPR
def amp_eval(y_true, y_pred, y_type, min_thres=3):
    if sum(y_true) < min_thres:
        return np.nan

    if y_type == 'aroc':
        return roc_auc_score(y_true, y_pred)

    else:
        return average_precision_score(y_true, y_pred)


plot_df = pd.DataFrame([{
    'type': d,
    'sample': c,
    'threshold': t,
    'aroc': amp_eval([int(gdsc_cnv.loc[g, c] > t) for g in genes], -df.loc[genes, c].values, 'aroc'),
    'aupr': amp_eval([int(gdsc_cnv.loc[g, c] > t) for g in genes], -df.loc[genes, c].values, 'aupr')
} for d, df in [('deseq2', c_deseq), ('mageck', c_mageck), ('bagel', c_bagel)] for c in samples for t in [0, 1]])

# Scatter
gs, pos = GridSpec(2, 2, hspace=.6, wspace=.6), 0
sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})

for x, y, t in [(i, j, t) for t in ['aroc', 'aupr'] for i, j in [('deseq2', 'mageck'), ('deseq2', 'bagel')]]:
    ax = plt.subplot(gs[pos])

    for thres in [0, 1]:
        xs = plot_df[(plot_df['type'] == x) & (plot_df['threshold'] == thres)].set_index('sample').loc[samples, t]
        ys = plot_df[(plot_df['type'] == y) & (plot_df['threshold'] == thres)].set_index('sample').loc[samples, t]

        ax.scatter(xs, ys, c=cmap[thres + 1], s=3, marker='x', lw=0.5)
        sns.rugplot(xs, height=.02, axis='x', c=cmap[thres + 1], lw=.3, ax=ax)
        sns.rugplot(ys, height=.02, axis='y', c=cmap[thres + 1], lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(t.upper())

    pos += 1

# handles = [mlines.Line2D([], [], colorfacecolor=cmap[t + 1], marker='x', label='>%s' % t, linewidth=0) for t in [1, 0]]
handles = [mpatches.Circle([.0, .0], .25, facecolor=cmap[t + 1], label='>%s' % t) for t in [1, 0]]
plt.legend(handles=handles, title='GISTIC', loc=4)

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/comparison_amplifications_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Boxplots
plot_df_boxplot = plot_df.copy()
plot_df_boxplot['threshold'] = plot_df_boxplot['threshold'] + 1

sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.boxplot('type', 'aroc', 'threshold', data=plot_df_boxplot, linewidth=.3, notch=True, fliersize=1, orient='v', palette=cmap)
plt.axhline(0.5, lw=.3, ls='-', color='#dfdfdf')
# plt.xticks([])
plt.setp(g.get_xticklabels(), rotation=90)
plt.ylabel('AROC')
plt.xlabel('')
plt.title('Amplifications')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='GISTIC')
plt.gcf().set_size_inches(.75, 2)
plt.savefig('reports/comparison_amplifications_boxplots_aroc.png', bbox_inches='tight', dpi=600)
plt.close('all')

sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.boxplot('type', 'aupr', 'threshold', data=plot_df_boxplot, linewidth=.3, notch=True, fliersize=1, orient='v', palette=cmap)
plt.axhline(0.0, lw=.3, ls='-', color='#dfdfdf')
# plt.xticks([])
plt.setp(g.get_xticklabels(), rotation=90)
plt.ylabel('AUPR')
plt.xlabel('')
plt.title('Amplifications')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='GISTIC')
plt.gcf().set_size_inches(.75, 2)
plt.savefig('reports/comparison_amplifications_boxplots_aupr.png', bbox_inches='tight', dpi=600)
plt.close('all')
