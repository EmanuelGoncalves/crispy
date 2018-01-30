#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import re
import igraph
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from matplotlib.colors import rgb2hex
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, uniform
from sklearn.metrics import roc_curve, auc


# - Import
# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# Crispr ~ Drug associations
lm_df = pd.read_csv('data/drug/lm_drug_crispr.csv')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

# Growth rate
growth = pd.read_csv('data/gdsc/growth_rate.csv', index_col=0)


# - Overlap
samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index))
print('Samples: %d' % len(samples))


# - PPI
# Drug samplesheet
ds = pd.read_csv('data/gdsc/drug_samplesheet.csv', index_col=0)

# Drug targets manual curated list
d_targets = ds['Target Curated'].dropna().to_dict()
d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}

# Biogrid
# net_i = igraph.Graph.Read_Pickle('data/resources/igraph_string.pickle')
net_i = igraph.Graph.Read_Pickle('data/resources/igraph_biogrid.pickle')
net_i_genes = set(net_i.vs['name'])
net_i_crispr = net_i_genes.intersection(set(lm_df['GENES']))

# Distance to drug target
d_targets_dist = pd.DataFrame({
    d: np.min(net_i.shortest_paths(source=d_targets[d].intersection(net_i_genes), target=net_i_crispr), axis=0) for d in d_targets if len(d_targets[d].intersection(net_i_genes)) > 0
}, index=net_i_crispr)


# Annotate associations
def drug_target_distance(d, t):
    dst = np.nan

    if (d in d_targets) and (t in d_targets[d]):
        dst = 0

    elif (d in d_targets_dist.columns) and (t in d_targets_dist.index):
        dst = d_targets_dist.loc[t, d]

    return dst


lm_df = lm_df.assign(target=[drug_target_distance(d, t) for d, t in lm_df[['DRUG_ID', 'GENES']].values])


# - Export table
g_info = pd.read_csv('data/resources/uniprot/uniprot-all.tab', sep='\t').dropna(subset=['Gene names', 'Function [CC]', 'Protein names'])
g_info['Gene names'] = [set(i.split(' ')) for i in g_info['Gene names']]

export_table = lm_df.query('lr_fdr < 0.05').sort_values('lr_fdr')
export_table['Protein names'] = [' // '.join(g_info[g_info['Gene names'].apply(lambda x: g in x)]['Protein names']) for g in export_table['GENES']]
export_table['Function'] = [' // '.join(g_info[g_info['Gene names'].apply(lambda x: g in x)]['Function [CC]']) for g in export_table['GENES']]
export_table[
    ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'GENES', 'Protein names', 'Function', 'beta', 'r2', 'lr_fdr', 'target']
].to_csv('data/drug/lm_drug_crispr_annotated.csv', index=False)


# - Volcano
plot_df = lm_df.copy()
plot_df['signif'] = ['*' if i < 0.05 else '-' for i in plot_df['lr_fdr']]
plot_df['log10_qvalue'] = -np.log10(plot_df['lr_pval'])

pal = dict(zip(*(['*', '-'], sns.light_palette(bipal_dbgd[0], 3, reverse=True).as_hex()[:-1])))

# Scatter
for i in ['*', '-']:
    plot_df_ = plot_df.query("(signif == '%s')" % i)
    plt.scatter(plot_df_['beta'], plot_df_['log10_qvalue'], edgecolor='white', lw=.1, s=3, alpha=.5, c=pal[i])

# Add FDR threshold lines
yy = plot_df.loc[(plot_df['lr_fdr'] - 0.05).abs().sort_values().index[0], 'lr_pval']
plt.text(plt.xlim()[0]*.98, -np.log10(yy) * 1.01, 'FDR 5%', ha='left', color=pal['*'], alpha=0.65, fontsize=5)
plt.axhline(-np.log10(yy), c=pal['*'], ls='--', lw=.3, alpha=.7)

# Add axis lines
plt.axvline(0, c=bipal_dbgd[0], lw=.1, ls='-', alpha=.3)

# Add axis labels and title
plt.title('Linear regression: Drug ~ CRISPR', fontsize=8, fontname='sans-serif')
plt.xlabel('Model coefficient (beta)', fontsize=8, fontname='sans-serif')
plt.ylabel('Log-ratio test p-value (-log10)', fontsize=8, fontname='sans-serif')

# Save plot
plt.gcf().set_size_inches(2., 4.)
plt.savefig('reports/drug/lmm_crispr_volcano.png', bbox_inches='tight', dpi=300)
plt.close('all')


# - Barplot count number of significant associations
plot_df = pd.DataFrame({
    'ypos': [0, 1], 'type': ['positive', 'negative'], 'count': [sum(lm_df.query('lr_fdr < 0.05')['beta'] > 0), sum(lm_df.query('lr_fdr < 0.05')['beta'] < 0)]
})

plt.barh(plot_df['ypos'], plot_df['count'], .8, color=bipal_dbgd[0], align='center')

for x, y in plot_df[['ypos', 'count']].values:
    plt.text(y - .5, x, str(y), color='white', ha='right' if y != 1 else 'center', va='center', fontsize=10)

plt.yticks(plot_df['ypos'])
plt.yticks(plot_df['ypos'], plot_df['type'])

plt.xlabel('# significant associations (FDR < 5%)')
plt.title('Significant Drug ~ CRISPR associations (%d)' % plot_df['count'].sum())

plt.gcf().set_size_inches(3, 1.5)
plt.savefig('reports/drug/signif_assoc_count.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Drug target Cumulative distributions
ax = plt.gca()

_, _, patches = ax.hist(
    lm_df.query('target != 0')['lr_pval'], lm_df.query('target != 0').shape[0],
    cumulative=True, normed=1, histtype='step', label='All', color=bipal_dbgd[0], lw=1.
)
patches[0].set_xy(patches[0].get_xy()[:-1])

_, _, patches = ax.hist(
    lm_df.query('target == 0')['lr_pval'], lm_df.query('target == 0').shape[0],
    cumulative=True, normed=1, histtype='step', label='Target', color=bipal_dbgd[1], lw=1.
)
patches[0].set_xy(patches[0].get_xy()[:-1])

_, _, patches = ax.hist(
    uniform.rvs(size=lm_df.shape[0]), lm_df.shape[0],
    cumulative=True, normed=1, histtype='step', label='Random', color='gray', lw=.3, ls='--'
)
patches[0].set_xy(patches[0].get_xy()[:-1])

plt.xlim(0, 1)
plt.ylim(0, 1)

ax.set_xlabel('Drug ~ CRISPR p-value')
ax.set_ylabel('Cumulative distribution')

ax.legend(loc=4)

plt.gcf().set_size_inches(2.5, 2)
plt.savefig('reports/drug/drug_target_cum_dist.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - PPI boxplot
plot_df = lm_df.dropna()
plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

order = ['Target', '1', '2', '3', '>=4']
palette = sns.light_palette(bipal_dbgd[0], len(order), reverse=True)

sns.boxplot('lr_pval', 'thres', data=plot_df, orient='h', linewidth=.3, notch=True, fliersize=1, palette=palette, order=order)

plt.ylabel('PPI distance')
plt.xlabel('Drug ~ Drug-target CRISPR p-value')

plt.title('Drug-targets in protein-protein networks')
plt.gcf().set_size_inches(3, 1.5)
plt.savefig('reports/drug/ppi_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - PPI significant associations
plot_df = lm_df.query('lr_fdr < 0.1').dropna()
plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

ax = plt.gca()
for t, c in zip(*(['Target', '1', '2', '3', '>=4'], [bipal_dbgd[1]] + list(map(rgb2hex, sns.light_palette(bipal_dbgd[0], len(order) - 1, reverse=True))))):
    fpr, tpr, _ = roc_curve((plot_df['thres'] == t).astype(int), 1 - plot_df['lr_fdr'])
    ax.plot(fpr, tpr, label='%s=%.2f (AUC)' % (t, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Protein-protein interactions\nDrug ~ CRISPR (FDR < 10%)')

legend = ax.legend(loc=4, prop={'size': 6})

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/drug/ppi_signif_roc.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
order = lm_df.query('lr_fdr < 0.05').groupby(['DRUG_NAME'])['lr_fdr'].min().sort_values()

plot_df = lm_df.query('lr_fdr < 0.05').copy().sort_values('lr_fdr')
plot_df = plot_df.groupby(['DRUG_NAME', 'GENES']).first().reset_index().sort_values('lr_fdr')
plot_df = plot_df.groupby('DRUG_NAME').head(10).set_index('DRUG_NAME')

plot_df__, n_top, xpos = [], 10, 0
for drug_name in order.iloc[10:20].index:
    plot_df_ = plot_df.loc[drug_name]
    plot_df_ = plot_df_.assign(y=-np.log10(plot_df_['lr_fdr']))

    plot_df_ = plot_df_.assign(xpos=np.arange(xpos, xpos + plot_df_.shape[0]))
    xpos += plot_df_.shape[0]

    plot_df__.append(plot_df_)

    xpos += 1

plot_df = pd.concat(plot_df__).reset_index()

# Plot
plt.bar(plot_df.query('target != 0')['xpos'], plot_df.query('target != 0')['y'], .8, color=bipal_dbgd[0], align='center')
plt.bar(plot_df.query('target == 0')['xpos'], plot_df.query('target == 0')['y'], .8, color=bipal_dbgd[1], align='center')

for x, y, t in plot_df[['xpos', 'y', 'target']].values:
    l = '-' if np.isnan(t) or np.isposinf(t) else ('T' if t == 0 else str(int(t)))
    plt.text(x, y - .25, l, color='white', ha='center', fontsize=6)

plt.ylim(ymax=4)

for k, v in plot_df.groupby('DRUG_NAME')['xpos'].mean().sort_values().to_dict().items():
    plt.text(v, 4 - .5, textwrap.fill(k, 15), ha='center', fontsize=6)

plt.xticks(plot_df['xpos'], plot_df['GENES'], rotation=90, fontsize=5)

plt.ylabel('Log-ratio FDR (-log10)')
plt.title('Top significant Drug ~ CRISPR associations')

plt.gcf().set_size_inches(8, 1.5)
plt.savefig('reports/drug/drug_associations_barplot.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Corrplot
idx = 14
d_id, d_name, d_screen, gene = lm_df.loc[idx, ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'GENES']].values

# Build data-frame
plot_df = pd.DataFrame({
    'drug': d_response.loc[(d_id, d_name, d_screen), samples],
    'crispr': crispr.loc[gene, samples],
}).dropna()
plot_df = pd.concat([plot_df, ss.loc[plot_df.index, ['Cancer Type', 'Microsatellite']]], axis=1)

# Plot
g = sns.jointplot(
    'crispr', 'drug', data=plot_df, kind='reg', space=0, color=bipal_dbgd[0], annot_kws=dict(stat='r'),
    marginal_kws={'kde': False}, joint_kws={'scatter_kws': {'edgecolor': 'w', 'lw': .3, 's': 12}, 'line_kws': {'lw': 1.}}
)

g.ax_joint.axhline(0, ls='-', lw=0.1, c=bipal_dbgd[0])
g.ax_joint.axvline(0, ls='-', lw=0.1, c=bipal_dbgd[0])

g.set_axis_labels('%s (fold-change)' % gene, '%s (screen=%s; ln IC50)' % (d_name, d_screen))

plt.gcf().set_size_inches(2., 2.)
plt.savefig('reports/drug/crispr_drug_corrplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
