#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy
import igraph
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd


# - Imports
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])
ds = pd.read_csv('data/gdsc/drug_samplesheet.csv', index_col=0)


# - Drug targets manual curated list
d_targets = ds['Target Curated'].dropna().to_dict()
d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}


# - Drug-Crispr associations
lmm_df = pd.read_csv('data/drug/lmm_drug_crispr.csv')


# - PPI
# Biogrid
net_i = igraph.Graph.Read_Pickle('data/resources/igraph_biogrid.pickle')
net_i_genes = set(net_i.vs['name'])
net_i_crispr = net_i_genes.intersection(set(lmm_df['crispr']))

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


lmm_df = lmm_df.assign(target=[drug_target_distance(d, t) for d, t in lmm_df[['drug_id', 'crispr']].values])


# - PPI boxplot
plot_df = lmm_df.dropna()
plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

order = ['Target', '1', '2', '3', '>=4']
palette = sns.light_palette(bipal_dbgd[0], len(order), reverse=True)


sns.boxplot('rank_beta', 'thres', data=plot_df, orient='h', linewidth=.3, notch=True, fliersize=1, palette=palette, order=order)

plt.ylabel('Distance')
plt.xlabel('Drug-target scaled rank p-value')

plt.title('Drug-targets in protein-protein networks')
plt.gcf().set_size_inches(2, 1.5)
plt.savefig('reports/drug/ppi_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
