#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import igraph
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from scipy.stats import pearsonr, iqr, uniform


# - Import
# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# Crispr ~ Drug associations
lm_df = pd.read_csv('data/drug/lm_drug_crispr.csv')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type', 'Microsatellite'])

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
net_i = igraph.Graph.Read_Pickle('data/resources/igraph_string.pickle')
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


# - Volcano
plot_df = lm_df.copy()
plot_df['signif'] = ['*' if i < 0.05 else '-' for i in plot_df['lr_fdr']]
plot_df['log10_qvalue'] = -np.log10(plot_df['lr_fdr'])

pal = dict(zip(*(['*', '-'], sns.light_palette(bipal_dbgd[0], 3, reverse=True).as_hex()[:-1])))

# Scatter
for i in ['*', '-']:
    plt.scatter(
        plot_df.query("(target != 0) & (signif == '%s')" % i)['beta'], plot_df.query("(target != 0) & (signif == '%s')" % i)['log10_qvalue'],
        edgecolor='white', lw=.1, s=5, alpha=.5, c=pal[i]
    )

plt.scatter(
    plot_df.query('target == 0')['beta'], plot_df.query('target == 0')['log10_qvalue'],
    edgecolor='white', lw=.1, s=5, alpha=.5, c=bipal_dbgd[1]
)

# Add FDR threshold lines
plt.text(plt.xlim()[0]*.98, -np.log10(.05) * 1.01, 'FDR 5%', ha='left', color=pal['*'], alpha=0.65, fontsize=5)
plt.axhline(-np.log10(.05), c=pal['*'], ls='--', lw=.3, alpha=.7)

# Add axis lines
plt.axvline(0, c=bipal_dbgd[0], lw=.1, ls='-', alpha=.3)

# Add axis labels and title
plt.title('Drug ~ CRISPR', fontsize=8, fontname='sans-serif')
plt.xlabel('Beta', fontsize=8, fontname='sans-serif')
plt.ylabel('q-value (-log10)', fontsize=8, fontname='sans-serif')

# Save plot
plt.gcf().set_size_inches(2., 4.)
plt.savefig('reports/drug/lmm_crispr_volcano.png', bbox_inches='tight', dpi=300)
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

# ax.set_title('Copy-number correlation with CRISPR bias')
ax.set_xlabel('Drug ~ CRISPR p-value')
ax.set_ylabel('Cumulative distribution')

ax.legend(loc='upper left')

plt.gcf().set_size_inches(2.5, 2)
plt.savefig('reports/drug/drug_target_cum_dist.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Corrplot
idx = 16

d_id, d_name, d_screen, gene = lm_df.loc[idx, ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'GENES']].values

plot_df = pd.DataFrame({
    'drug': d_response.loc[(d_id, d_name, d_screen), samples],
    'crispr': crispr.loc[gene, samples],
}).dropna()
plot_df = pd.concat([plot_df, ss.loc[plot_df.index, ['Cancer Type', 'Microsatellite']]], axis=1)


def marginal_distplot(a, vertical=False, **kws):
    x = plot_df['drug'] if vertical else plot_df['crispr']
    ax = sns.distplot(x, vertical=vertical, **kws)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_ylim((min(x) * 1.05, max(x) * 1.05)) if vertical else ax.set_xlim((min(x) * 1.05, max(x) * 1.05))


sns.set(
    style='ticks', context='paper', font_scale=.75,
    rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5}
)
g = sns.JointGrid('crispr', 'drug', plot_df, space=0, ratio=8)

sns.regplot(
    x='crispr', y='drug', data=plot_df, ax=g.ax_joint,
    color=bipal_dbgd[0], truncate=True, fit_reg=True,
    scatter_kws={'s': 10, 'edgecolor': 'w', 'linewidth': .3, 'alpha': .8}, line_kws={'linewidth': .5}
)

g.plot_marginals(marginal_distplot, color=bipal_dbgd[0], kde=False)

g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}')

g.ax_joint.axhline(0, ls='-', lw=0.1, c=bipal_dbgd[0])
g.ax_joint.axvline(0, ls='-', lw=0.1, c=bipal_dbgd[0])

g.set_axis_labels('%s (fold-change)' % gene, '%s (ln IC50)' % d_name)

plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/drug/crispr_drug_corrplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
