#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from natsort import natsorted
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec
from sklearn.decomposition.pca import PCA
from crispy.data_matrix import DataMatrix
from crispy.biases_correction import CRISPRCorrection
from crispy.benchmark_plot import plot_cumsum_auc, plot_chromosome, plot_cnv_rank


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0)

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential genes')

# samplesheet
control = 'Plasmid_v1.1'
samplesheet = pd.read_csv('data/gdsc/crispr_drug/HT29_Dabraf_samplesheet.csv', index_col='sample').replace(np.nan, 'Initial')

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)[['HT-29']]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))


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
print(fold_changes.shape)


# - Plot
cmaps = {
    f: dict(zip(*(natsorted(set(samplesheet[f])), sns.color_palette(c, len(set(samplesheet[f]))).as_hex())))
    for f, c in [('replicate', 'Greys'), ('medium', 'Blues'), ('time', 'Reds')]
}

fold_changes_gene = fold_changes.groupby(sgrna_lib.loc[fold_changes.index, 'GENES']).mean()

# Essential genes AROCs
ax, ax_stats = plot_cumsum_auc(fold_changes_gene, essential, legend=[], plot_mean=True, cmap='GnBu')
plt.title('Essential genes AROC')
plt.gcf().set_size_inches(2, 2)
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
g.ax_heatmap.legend(handles=handles, title='Tissue', loc='center left', bbox_to_anchor=(1.05, 0.5))
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
plt.savefig('reports/HT29_dabraf_replicates_boxplot_gene.png', bbox_inches='tight', dpi=600)
plt.close('all')

# PCA
n_components = 5
fc_pca = PCA(n_components=n_components).fit(fold_changes_gene.T)
print(fc_pca.explained_variance_ratio_)

fc_pcs = pd.DataFrame(fc_pca.transform(fold_changes_gene.T), index=fold_changes_gene.columns, columns=range(n_components))
fc_pcs = pd.concat([fc_pcs, samplesheet], axis=1)

medium_markers = [('Initial', 'o'), ('DMSO', 'X'), ('Dabraf', 'P')]

fig, gs = plt.figure(figsize=(3.8, 1.5)), GridSpec(1, 2, hspace=.4, wspace=.4)
for i, (pc_x, pc_y) in enumerate([(0, 1), (0, 2)]):
    ax = plt.subplot(gs[i])

    for mdm, mrk in medium_markers:
        plot_df = fc_pcs.query("medium == '%s'" % mdm)
        ax.scatter(
            plot_df[pc_x], plot_df[pc_y], c=plot_df['time'].replace(cmaps['time']), marker=mrk, lw=.3, edgecolor='white'
        )

    ax.set_xlabel('PC%d (%.1f%%)' % (pc_x + 1, fc_pca.explained_variance_ratio_[pc_x] * 100))
    ax.set_ylabel('PC%d (%.1f%%)' % (pc_y + 1, fc_pca.explained_variance_ratio_[pc_y] * 100))

handles = [mlines.Line2D([], [], .25, color='k', marker=mrk, label=mdm, lw=0) for mdm, mrk in medium_markers] + \
          [mlines.Line2D([], [], .25, color=cmaps['time'][d], marker='o', label=d.replace('D', 'Day '), lw=0) for d in natsorted(cmaps['time'])]
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.savefig('./reports/HT29_dabraf_replicates_pca_gene.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - CNV correction
corrected_fc = {}

for sample in fold_changes:
    # Build data-frame
    df = pd.concat([fold_changes[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
    df = df[df['GENES'].isin(cnv_abs.index)]
    df = df.assign(cnv=list(cnv_abs.loc[df['GENES'], samplesheet.loc[sample, 'cell']]))

    # Correction
    df_gene = df.groupby('GENES').agg({'logfc': np.mean, 'cnv': np.mean, 'CHRM': 'first'})

    crispy = CRISPRCorrection().rename(sample).fit_by(by=df_gene['CHRM'], X=df_gene[['cnv']], y=df_gene['logfc'])
    df_corrected = pd.concat([
        v.to_dataframe(df.query("CHRM == '%s'" % k)[['cnv']], df.query("CHRM == '%s'" % k)['logfc'], return_var_exp=True) for k, v in crispy.items()
    ])
    corrected_fc[sample] = df_corrected.assign(GENES=sgrna_lib.loc[df_corrected.index, 'GENES']).assign(sample=sample)

crispy_sgrna_df = pd.DataFrame({s: corrected_fc[s]['regressed_out'] for s in fold_changes})
crispy_sgrna_df.to_csv('data/gdsc/crispr_drug/HT29_dabraf_foldchanges_corrected.csv')
print(crispy_sgrna_df.shape)
