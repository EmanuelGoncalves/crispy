#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import re
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from scipy_sugar.stats import quantile_gaussianize
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# -- Modify to True to generate preprocessing plots
DO_PLOTS = False
# --

# - Import cell lines colour palette
cmap = pd.Series.from_csv('data/gdsc/colour_palette_cell_lines.csv')


# - Import list of approved files
approved_files = pd.read_csv('data/gdsc/crispr/crispr_raw_counts_files_vFB.csv', index_col=0)
approved_files = {i.split('.')[0] for i in approved_files[approved_files['to_use'] == 'Yes'].index}


# - Import gene lists
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])
non_essential = list(pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'])


# - Read cell lines manifest
manifest = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t')
manifest['CGaP_Sample_Name'] = [i.split('_')[0] for i in manifest['CGaP_Sample_Name']]
manifest = manifest.groupby('CGaP_Sample_Name')['Cell_Line_Name'].first()


# - Read mageck fold-changes files
d_files = {}
for x in [f for f in os.listdir('data/gdsc/crispr/mageck/') if f.endswith('.sgrna_summary.txt') and f.split('.')[0] in approved_files]:
    d_files.setdefault(x.split('_')[0], []).append(x)


# - Import sgRNA fold-changes and preprocesses
fc_mean, ess_auc = {}, {}
# c = 'A375'
for c in d_files:
    # Read samples replicates
    df = pd.DataFrame({f.split('.')[0]: pd.read_csv('./data/gdsc/crispr/mageck/%s' % f, sep='\t', index_col=0)['LFC'] for f in d_files[c]}).dropna()

    # Consider Library v1.1 sgRNAs
    df.index = [i[2:] if re.match('sg', i) else i for i in df.index]

    # Boxplot
    if DO_PLOTS:
        sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
        g = sns.boxplot(df, linewidth=.3, notch=True, fliersize=1, orient='v', color='#34495e')
        plt.axhline(0, lw=.3, ls='-', color='#dfdfdf')
        plt.setp(g.get_xticklabels(), rotation=90)
        plt.ylabel('Fold-change (sgRNA)')
        plt.xlabel('Samples')
        plt.gcf().set_size_inches(.1 * df.shape[1], 1.5)
        plt.savefig('data/gdsc/crispr/mageck_plots/%s_sgrna_boxplots.png' % manifest[c], bbox_inches='tight', dpi=600)
        plt.close('all')

    # Average sgRNAs per gene
    df = df.groupby([i.split('_')[0] for i in df.index]).mean()

    # Evalute
    df_ess_auc = {c: roc_auc_score([int(i in essential) for i in df.index], -df[c].values) for c in df}
    df_ess_apr = {c: average_precision_score([int(i in essential) for i in df.index], -df[c].values) for c in df}

    ess_auc[manifest[c]] = {'aroc': df_ess_auc, 'apr': df_ess_apr}
    print('[%s] Essential AUC: ' % dt.now().strftime('%Y-%m-%d %H:%M:%S'), '; '.join(['%s: %.2f' % (k, v) for k, v in df_ess_auc.items()]))
    print('[%s] Essential APR: ' % dt.now().strftime('%Y-%m-%d %H:%M:%S'), '; '.join(['%s: %.2f' % (k, v) for k, v in df_ess_apr.items()]))

    # Corrplot
    if DO_PLOTS:
        def corrfunc(x, y, **kws):
            r, p = pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(
                'R={:.2f}\n{:s}'.format(r, ('p<1e-4' if p < 1e-4 else ('p=%.1e' % p))),
                xy=(.5, .5), xycoords=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=7
            )

        sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
        g = sns.PairGrid(df, despine=False, palette=['#34495e'])
        g.map_upper(sns.regplot, color='#34495e', truncate=True, scatter_kws={'s': 1, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .8}, line_kws={'linewidth': .5})
        g.map_diag(sns.distplot, kde=False, bins=30, hist_kws={'color': '#34495e'})
        g.map_lower(corrfunc)
        plt.gcf().set_size_inches(2., 2., forward=True)
        plt.savefig('data/gdsc/crispr/mageck_plots/%s_sgrna_corrplot.png' % manifest[c], bbox_inches='tight', dpi=600)
        plt.close('all')

    # Average replicates
    df = df.mean(1)

    # Store result
    fc_mean[manifest[c]] = df.copy()
    print('[%s] %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c))

fc_mean = pd.DataFrame(fc_mean).dropna()


# - Gene essential AUC
# Area under de ROC curve
plot_df = pd.DataFrame([{'cell_line': c, 'replicate': s, 'aroc': ess_auc[c]['aroc'][s]} for c in ess_auc for s in ess_auc[c]['aroc']])

sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
sns.boxplot(plot_df['aroc'], fliersize=1, linewidth=.3, color='#34495e', orient='v')

for x, y in plot_df.sort_values('aroc').head(3)[['aroc', 'replicate']].values:
    plt.scatter(0, x, s=3, marker='x', c='#e74c3c', lw=0.5)
    plt.annotate(y, (0, x), fontsize=3)

plt.ylabel('Essential genes (AROC)')
plt.gcf().set_size_inches(.2, 2., forward=True)
plt.savefig('data/gdsc/crispr/mageck_plots/essential_genes_aroc_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Area under the precission recall curve
plot_df = pd.DataFrame([{'cell_line': c, 'replicate': s, 'aupr': ess_auc[c]['apr'][s]} for c in ess_auc for s in ess_auc[c]['apr']])

sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
sns.boxplot(plot_df['aupr'], fliersize=1, linewidth=.3, color='#34495e', orient='v')

for x, y in plot_df.sort_values('aupr').head(3)[['aupr', 'replicate']].values:
    plt.scatter(0, x, s=3, marker='x', c='#e74c3c', lw=0.5)
    plt.annotate(y, (0, x), fontsize=3)

plt.ylabel('Essential genes (AUPR)')
plt.gcf().set_size_inches(.2, 2., forward=True)
plt.savefig('data/gdsc/crispr/mageck_plots/essential_genes_aupr_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Test AROC of essential genes
c = 'MCF7'
y_true, probas_pred = np.array([int(i in essential) for i in fc_mean[c].index]), -fc_mean[c].values

sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
curve_recall, curve_precision, _ = precision_recall_curve(y_true, probas_pred)
plt.plot(curve_recall, curve_precision, label='%s (AUPR %0.2f)' % (c, average_precision_score(y_true, probas_pred)), c='#34495e', lw=1.)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.legend(loc='lower right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Essential genes')
plt.gcf().set_size_inches(2, 2)
plt.savefig('data/gdsc/crispr/mageck_plots/essential_genes_aupr_curve_%s.png' % c, bbox_inches='tight', dpi=600)
plt.close('all')


# - Boxplots of essential/non-essential genes
order = list(cmap[list(fc_mean)].sort_values().index)
hue_order = ['Other', 'Non-essential', 'Essential']

pal = {'Other': '#dfdfdf', 'Non-essential': '#34495e', 'Essential': '#e74c3c'}

plot_df = fc_mean.unstack().reset_index()
plot_df.columns = ['sample', 'gene', 'score']
plot_df['type'] = ['Essential' if g in essential else ('Non-essential' if g in non_essential else 'Other') for g in plot_df['gene']]

sns.set(style='ticks', context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.boxplot('sample', 'score', 'type', data=plot_df, linewidth=.3, notch=True, fliersize=1, orient='v', order=order, hue_order=hue_order, palette=pal)
plt.axhline(0, lw=.3, ls='-', color='#dfdfdf')
sns.despine()
# plt.xticks([])
plt.setp(g.get_xticklabels(), rotation=90)
plt.ylabel('fold-changes')
plt.xlabel('Samples')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(12, 2)
plt.savefig('reports/mageck_normalised_boxplots.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Export matrix
fc_mean.to_csv('data/gdsc/crispr/mageck_gene_fold_changes_mean.csv')
# fc_mean = pd.read_csv('data/gdsc/crispr/mageck_gene_fold_changes_mean.csv', index_col=0)
