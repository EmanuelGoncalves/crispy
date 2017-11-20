#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from natsort import natsorted
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, squareform
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score, silhouette_samples, calinski_harabaz_score
from sklearn.neighbors.classification import KNeighborsClassifier
from crispy.data_matrix import DataMatrix
from sklearn.decomposition.pca import PCA
from crispy.biases_correction import CRISPRCorrection
from crispy.benchmark_plot import plot_cumsum_auc, plot_chromosome, plot_cnv_rank
from scipy.stats.distributions import hypergeom


def read_gmt(file_path):
    with open(file_path) as f:
        signatures = {l.split('\t')[0]: set(l.strip().split('\t')[2:]) for l in f.readlines()}
    return signatures


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0)

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential genes')

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/gene_expression/non_expressed_genes.pickle', 'rb'))

# samplesheet
control = 'Plasmid_v1.1'
samplesheet = pd.read_csv('data/gdsc/crispr_drug/HT29_Dabraf_samplesheet.csv', index_col='sample')
samplesheet = samplesheet.assign(id=samplesheet['medium'] + '_' + samplesheet['time'])

# # Copy-number absolute counts
# cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)[['HT-29']]
# cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))


# - Calculate fold-changes
# Import read counts
raw_counts = {}
for f in samplesheet['file']:
    raw_counts.update(pd.read_csv('data/gdsc/crispr_drug/ht29_dabraf/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict())

# Build counts data-frame (discard sgRNAs with problems)
qc_failed_sgrnas = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]
raw_counts = DataMatrix(raw_counts).drop(qc_failed_sgrnas, errors='ignore')

# Filter by minimum required control counts
raw_counts = raw_counts[raw_counts[control] >= 30]

# Build contrast matrix
design = raw_counts.drop(control, axis=1).columns
design = pd.get_dummies(design).set_index(design).T.assign(control=-1).rename(columns={'control': control}).T

# Estimate fold-changes
fold_changes = raw_counts.add_constant(1).counts_normalisation().log_transform().estimate_fold_change(design).zscore()

# Export
fold_changes.to_csv('data/gdsc/crispr_drug/HT29_dabraf_foldchanges.csv')

fold_changes_gene = fold_changes.groupby(sgrna_lib.loc[fold_changes.index, 'GENES']).mean()
fold_changes_gene.to_csv('data/gdsc/crispr_drug/HT29_dabraf_foldchanges_gene.csv')
print(fold_changes.shape)

# fold_changes_gene = pd.read_csv('data/gdsc/crispr_drug/HT29_dabraf_foldchanges_gene.csv', index_col=0)

# - Plot
cmaps = {
    f: dict(zip(*(natsorted(set(samplesheet[f])), sns.color_palette(c, len(set(samplesheet[f]))).as_hex())))
    for f, c in [('replicate', 'Greys'), ('medium', 'Blues'), ('time', 'Reds')]
}

# Essential genes AROCs
ax, ax_stats = plot_cumsum_auc(fold_changes_gene, essential, legend=[], plot_mean=True, cmap='GnBu')
plt.title('Essential genes')
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/HT29_dabraf_essential_arocs.pdf', bbox_inches='tight')
plt.savefig('reports/HT29_dabraf_essential_arocs.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Corrplot
plot_df = fold_changes_gene.corr()

crmaps = samplesheet.loc[plot_df.index, ['replicate', 'medium', 'time']].replace(cmaps)

g = sns.clustermap(
    plot_df, cmap='GnBu', lw=.3, figsize=(7, 7), col_colors=crmaps, row_colors=crmaps,
    method='complete', metric='correlation', yticklabels=False, xticklabels=False
)

handles = [mpatches.Circle([.0, .0], .25, facecolor=cmaps[f][v], label=v) for f in cmaps for v in cmaps[f]]
g.ax_heatmap.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.savefig('reports/HT29_dabraf_replicates_clustermap_gene.pdf', bbox_inches='tight')
plt.savefig('reports/HT29_dabraf_replicates_clustermap_gene.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Boxplot
smap = (samplesheet['medium'] + '_' + samplesheet['time']).to_dict()

plot_df = fold_changes_gene.corr().rename(index=smap, columns=smap)
plot_df = plot_df.where(np.triu(np.ones(plot_df.shape), 1).astype(np.bool)).unstack().dropna().reset_index().rename(columns={'level_0': 'S1', 'level_1': 'S2', 0: 'cor'})
plot_df = plot_df.assign(replicate=['Yes' if s1 == s2 else 'No' for s1, s2 in plot_df[['S1', 'S2']].values])

sns.boxplot('replicate', 'cor', data=plot_df, fliersize=0, palette='Reds_r', linewidth=.5)
sns.swarmplot('replicate', 'cor', data=plot_df, size=2, palette='Reds_r', linewidth=.3, edgecolor='white')
plt.xlabel('Replicate')
plt.ylabel('Pearson\'s R')
plt.gcf().set_size_inches(1, 3)
plt.savefig('reports/HT29_dabraf_replicates_boxplot_gene.pdf', bbox_inches='tight')
plt.savefig('reports/HT29_dabraf_replicates_boxplot_gene.png', bbox_inches='tight', dpi=600)
plt.close('all')

# PCA
n_components = 5
fc_pca = PCA(n_components=n_components).fit(fold_changes_gene.T)

fc_pcs = pd.DataFrame(fc_pca.transform(fold_changes_gene.T), index=fold_changes_gene.columns, columns=range(n_components))
fc_pcs = pd.concat([fc_pcs, samplesheet], axis=1)

t_cmap = pd.Series(dict(zip(*(natsorted(set(samplesheet['time'])), sns.light_palette('#37454B', n_colors=len(set(samplesheet['time'])) + 1).as_hex()[1:]))))

medium_markers = [('Initial', 'o'), ('DMSO', 'X'), ('Dabraf', 'P')]

fig, gs = plt.figure(figsize=(3.8, 1.5)), GridSpec(1, 2, hspace=.4, wspace=.4)
for i, (pc_x, pc_y) in enumerate([(0, 1)]):
    ax = plt.subplot(gs[i])

    for mdm, mrk in medium_markers:
        plot_df = fc_pcs.query("medium == '%s'" % mdm)
        ax.scatter(
            plot_df[pc_x], plot_df[pc_y], c=t_cmap[plot_df['time']], marker=mrk, lw=.3, edgecolor='white'
        )

    ax.set_xlabel('PC%d (%.1f%%)' % (pc_x + 1, fc_pca.explained_variance_ratio_[pc_x] * 100))
    ax.set_ylabel('PC%d (%.1f%%)' % (pc_y + 1, fc_pca.explained_variance_ratio_[pc_y] * 100))

    ax.axhline(0, ls='-', c='k', lw=.1, alpha=.5)
    ax.axvline(0, ls='-', c='k', lw=.1, alpha=.5)

handles = [mlines.Line2D([], [], .25, color='#37454B', marker=mrk, label=mdm, lw=0) for mdm, mrk in medium_markers] + \
          [mlines.Line2D([], [], .25, color=t_cmap[d], marker='o', label=d.replace('D', 'Day '), lw=0) for d in natsorted(t_cmap.index)]
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.savefig('reports/HT29_dabraf_replicates_pca_gene.pdf', bbox_inches='tight')
plt.savefig('reports/HT29_dabraf_replicates_pca_gene.png', bbox_inches='tight', dpi=600)
plt.close('all')

# # Copy-number effect in non-expressed genes
# plot_df = pd.concat([fold_changes_gene, cnv_abs['HT-29'].rename('cnv')], axis=1).dropna().query('cnv != -1')
#
# order = range(int(max(plot_df['cnv'])) + 1)
# plot_cnv_rank(plot_df['cnv'], plot_df.drop('cnv', axis=1).mean(1).rename('Average'), stripplot=False, order=order)
# ax.set_title('CNV effect non-expressed genes')
# plt.gcf().set_size_inches(3, 2)
# plt.savefig('reports/HT29_dabraf_cnv_effect.png', bbox_inches='tight', dpi=600)
# plt.close('all')

# - Comparison
fc_limma = pd.read_csv('data/gdsc/crispr_drug/HT29_dabraf_foldchanges_gene_limma.csv', index_col=0)

conditions = [('DMSO', '#37454B'), ('Dabraf', '#F2C500')]

# df = fc_limma.loc[X_cls.query('cls == 1').index]
# df = fc_limma.loc[[i for i in fc_limma.index if i.startswith('TSC')]]
df = fc_limma.head(12)

nrows, ncols = int(np.ceil(df.shape[0] / 3)), np.min([df.shape[0], 3])
fig, gs = plt.figure(figsize=(4.7 * ncols, 3 * nrows)), GridSpec(nrows, ncols, hspace=.5, wspace=.3)

for i, g in enumerate(df.index):
    ax = plt.subplot(gs[i])

    plot_df = pd.concat([fold_changes_gene.loc[g], samplesheet], axis=1)
    plot_df = pd.concat([
        plot_df.query("medium != 'Initial'"),
        plot_df.query("medium == 'Initial'").replace({'medium': {'Initial': 'Dabraf'}}),
        plot_df.query("medium == 'Initial'").replace({'medium': {'Initial': 'DMSO'}}),
    ])

    for mdm, clr in conditions:
        plot_df_mrk = plot_df.query("medium == '%s'" % mdm)
        plot_df_mrk = plot_df_mrk.assign(time=plot_df_mrk['time'].apply(lambda x: int(x[1:])))
        plot_df_mrk = plot_df_mrk.groupby('time')[g].agg([min, max, np.mean])
        plot_df_mrk = plot_df_mrk.assign(time=['D%d' % i for i in plot_df_mrk.index])

        ax.errorbar(
            plot_df_mrk.index, plot_df_mrk['mean'], c=clr, fmt='--o', label=mdm, capthick=0.5, capsize=2, elinewidth=1,
            yerr=[plot_df_mrk['mean'] - plot_df_mrk['min'], plot_df_mrk['max'] - plot_df_mrk['mean']]
        )

    ax.axhline(0, ls='-', c='k', lw=.1, alpha=.5)

    ax.set_xticks(list(plot_df_mrk.index))

    ax.set_xlabel('Days')
    ax.set_ylabel('CRISPR fold-change')
    ax.legend(title='Condition')
    ax.set_title('%s\n(F-statistics q-value=%.1e)' % (g, fc_limma.loc[g, 'adj.P.Val']))

# handles = [mlines.Line2D([], [], .25, color=c, marker='o', label=l, lw=0) for l, c in conditions]
# plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))

plt.savefig('reports/HT29_dabraf_pointplot_subset.pdf', bbox_inches='tight')
# plt.savefig('reports/HT29_dabraf_pointplot_subset.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
X_cols = samplesheet.query("medium != 'DMSO'").groupby('id')['time'].first().reset_index().set_index('time')['id']

X = fold_changes_gene.loc[fc_limma[fc_limma['adj.P.Val'] < 0.1].index].rename(columns=samplesheet['id'])
X = X.groupby(X.columns, axis=1).mean()[X_cols[natsorted(X_cols.index)].values]
X = X.drop(['ATP6V0C', 'PSMB4'], errors='ignore')


# -
Z = linkage(X, 'average')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')

dendrogram(
    Z,
    # truncate_mode='lastp',
    p=12,
    # show_leaf_counts=False,
    # show_contracted=True,
    no_labels=True,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=12.,  # font size for the x axis labels
)
plt.savefig('reports/HT29_dabraf_clustering_dendogram.png', bbox_inches='tight', dpi=600)
plt.close('all')

last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev, c='#F2C500')

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev, c='#37454B')

k = acceleration_rev.argmax() + 2
plt.axvline(k, c='#37454B', lw=.3)
plt.title('Clusters: %d' % (k))

plt.savefig('reports/HT29_dabraf_clustering_elbow.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
clusters = fcluster(Z, k, criterion='maxclust')

#
X_cls = X.assign(cls=clusters)

n_clusters = len(set(X_cls['cls']))

fig, gs = plt.figure(figsize=(3 * n_clusters, 2)), GridSpec(1, n_clusters, hspace=.2, wspace=.2)
for idx, n in enumerate(set(X_cls['cls'])):
    ax = plt.subplot(gs[idx])

    df = X_cls.query("cls == %d" % n).drop('cls', axis=1)

    for yy in df.values:
        ax.plot([8, 10, 14, 18, 21], yy, ls='-', c='#F2C500', alpha=.2)

    ax.plot([8, 10, 14, 18, 21], df.mean().values, ls='-', c='#37454B')

    ax.set_title('#genes=%d (EGFR=%r)' % (df.shape[0], 'EGFR' in df.index))

plt.savefig('reports/HT29_dabraf_clustering_tsplot.png', bbox_inches='tight', dpi=600)
plt.close('all')



# -
signature = set(X_cls.query("cls == %d" % X_cls.loc['EGFR', 'cls']).index)

go_terms = {
    'BP': read_gmt('data/resources/msigdb/c5.bp.v6.0.symbols.gmt'),
    'CC': read_gmt('data/resources/msigdb/c5.cc.v6.0.symbols.gmt'),
    'MF': read_gmt('data/resources/msigdb/c5.mf.v6.0.symbols.gmt'),
    'HL': read_gmt('data/resources/msigdb/h.all.v6.0.symbols.gmt'),
    'KEGG': read_gmt('data/resources/msigdb/c2.cp.kegg.v6.0.symbols.gmt')
}


def hypergeom_test(signature, background, sublist):
    """
    Performs hypergeometric test

    Arguements:
            signature: {string} - Signature IDs
            background: {string} - Background IDs
            sublist: {string} - Sub-set IDs

    # hypergeom.sf(x, M, n, N, loc=0)
    # M: total number of objects,
    # n: total number of type I objects
    # N: total number of type I objects drawn without replacement

    """
    pvalue = hypergeom.sf(
        len(sublist.intersection(signature)),
        len(background),
        len(background.intersection(signature)),
        len(sublist)
    )

    intersection = len(sublist.intersection(signature).intersection(signature))

    return pvalue, intersection


henrch = pd.DataFrame([{
    'db': db,
    'pathway': path,
    'hyper': hypergeom_test(signature, set(X.index), go_terms[db][path])
} for db in go_terms for path in go_terms[db]]).dropna()
henrch = henrch.assign(pvalue=[i[0] for i in henrch['hyper']]).assign(intersection=[i[1] for i in henrch['hyper']]).drop('hyper', axis=1)
henrch = henrch.query("intersection >= 5").sort_values('pvalue')
print(henrch.head(50))


# -

