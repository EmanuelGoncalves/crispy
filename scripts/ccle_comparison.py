#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats.stats import zscore
from crispy.biases_correction import CRISPRCorrection
from crispy.benchmark_plot import plot_cumsum_auc, plot_chromosome
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern, PairwiseKernel

# - Gene lists
essential = pd.read_csv('data/ccle/crispr_ceres/Hart_217_essential_genes.txt', sep='\t', header=None)[0].rename('essential')

# - CCLE: log fold-changes
sgrna_lib = pd.read_csv('data/ccle/crispr_ceres/sgrnamapping.csv', index_col=0).dropna()
sgrna_lib_count = pd.Series(dict(zip(*(np.unique(sgrna_lib.index, return_counts=True)))))
sgrna_lib = sgrna_lib.loc[sgrna_lib_count[sgrna_lib_count == 1].index]
sgrna_lib = sgrna_lib.assign(chr=[i.split('_')[0] for i in sgrna_lib['Locus']])
sgrna_lib = sgrna_lib.assign(pos=[int(i.split('_')[1]) for i in sgrna_lib['Locus']])

samplesheet = pd.read_csv('data/ccle/crispr_ceres/replicatemap.csv', index_col=0).dropna()

crispr_sg = pd.read_csv('data/ccle/crispr_ceres/sgrnareplicateslogfcnorm.csv', index_col=0).loc[sgrna_lib.index, samplesheet.index]
crispr_sg = crispr_sg.groupby(samplesheet.loc[crispr_sg.columns, 'CellLine'], axis=1).mean()

crispr_std = crispr_sg.groupby(sgrna_lib.loc[crispr_sg.index, 'Gene'], axis=0).std()
crispr = crispr_sg.groupby(sgrna_lib.loc[crispr_sg.index, 'Gene'], axis=0).mean()

# - Library parameters
avana_hyper = pd.read_csv('data/ccle/crispr_ceres/avana_guide_activity.tsv', sep='\t', index_col=0)

# - CCLE: ceres
ceres = pd.read_csv('data/ccle/crispr_ceres/ceresgeneeffects.csv', index_col=0)

# - Copy-number
cnv_seg = pd.read_csv('data/ccle/crispr_ceres/CCLE_copy_number.seg.txt', sep='\t')
cnv = pd.read_csv('data/ccle/crispr_ceres/data_log2CNA.txt', sep='\t', index_col=0)
gistic = pd.read_csv('data/ccle/crispr_ceres/data_CNA.txt', sep='\t', index_col=0).drop(['Entrez_Gene_Id', 'Cytoband'], axis=1)


# -
samples = list(set(crispr).intersection(cnv))
genes = list(set(crispr.index).intersection(ceres.index).intersection(cnv.index))

sample = 'SW403_LARGE_INTESTINE'
df_corrected = {}
for sample in samples:
    df = pd.concat([
        crispr.loc[genes, sample].rename('logfc'),
        ceres.loc[genes, sample].rename('ceres'),
        sgrna_lib.groupby('Gene')['chr'].first().loc[genes].rename('chr'),
        sgrna_lib.groupby('Gene')['pos'].mean().loc[genes].rename('pos') / 1e6,
        2 * (2 ** cnv.loc[genes, sample].rename('cnv'))
    ], axis=1)

    crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['logfc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

    df_corrected[sample] = crispy['logfc'] - crispy['k_mean'] + crispy['mean']
df_corrected = pd.DataFrame(df_corrected)


# -
cnv_auc = {}
for sample in df_corrected:
    plot_df = pd.concat([
        crispr.loc[genes, sample].rename('logfc'),
        ceres.loc[genes, sample].rename('ceres'),
        df_corrected.loc[df.index, sample].rename('crispy'),
    ], axis=1)

    _, ax_stats = plot_cumsum_auc(plot_df, essential[essential.isin(genes)])
    # _, ax_stats = plot_cumsum_auc(plot_df, gistic.loc[gistic[sample] > 1, sample].index)
    plt.close('all')

    cnv_auc[sample] = ax_stats['auc']
cnv_auc = pd.DataFrame(cnv_auc)
print(cnv_auc)

#
plot_df = pd.concat([
    crispr.loc[genes, sample].rename('logfc'),
    ceres.loc[genes, sample].rename('ceres'),
    crispy['k_mean'].rename('crispy'),
    2 * (2 ** cnv.loc[genes, sample].rename('cnv')),
    sgrna_lib.groupby('Gene')['chr'].first().loc[genes].rename('chr'),
    sgrna_lib.groupby('Gene')['pos'].mean().loc[genes].rename('pos'),
], axis=1)

plot_df = plot_df.query("chr == 'chr6'")

ax = plot_chromosome(plot_df['pos'], plot_df['logfc'], plot_df['crispy'], highlight=['RPS3A'])
ax.scatter(plot_df['pos'], plot_df['cnv'], s=4, marker='.', lw=0, c='#b1b1b1', alpha=.9)
plt.gcf().set_size_inches(4, 2)
plt.savefig('reports/test_chrplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
plot_df = pd.concat([
    crispr[sample].rename('logfc'),
    ceres[sample].rename('ceres'),
    df_corrected[sample].rename('crispy'),
], axis=1).dropna()
plot_df = plot_df.assign(hue=plot_df.index.isin(
    pd.Series(dict(zip(*(plot_df.index, zscore(plot_df['ceres']) - zscore(plot_df['crispy']))))).sort_values().tail(5).index
).astype(str))

sns.pairplot(plot_df, 'hue')
