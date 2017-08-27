#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])
non_genes = pickle.load(open('data/gdsc/gene_expression/non_expressed_genes.pickle', 'rb'))

# samplesheet
ss = pd.read_csv('data/gdsc/crispr/crispr_raw_counts_files_vFB.csv', sep=',', index_col=0).query("to_use == 'Yes'")

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna(how='all').dropna(how='all', axis=1)

# Read cell lines manifest
manifest = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t')
manifest['CGaP_Sample_Name'] = [i.split('_')[0] for i in manifest['CGaP_Sample_Name']]
manifest = manifest.groupby('CGaP_Sample_Name')['Cell_Line_Name'].first()

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])


# -
# Assemble counts matrix
c_sample = 'HT29'
c_files = set(ss[[i.split('_')[0] == c_sample for i in ss.index]].index)

c_counts = {}
for f in c_files:
    df = pd.read_csv('data/gdsc/crispr/raw_counts/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict()
    c_counts.update(df)

c_counts = pd.DataFrame(c_counts).drop('sgPOLR2K_1')

c_counts += 1

# Estimate samples median-of-ratios
c_counts_gmean = pd.Series(st.gmean(c_counts, axis=1), index=c_counts.index)
c_counts_gmean = c_counts.T.divide(c_counts_gmean).T
c_counts_gmean = c_counts_gmean.median()

c_counts = c_counts.divide(c_counts_gmean)

# Define control sample
c_control = [i for i in c_counts if i in ['ERS717283.plasmid', 'CRISPR_C6596666.sample']][0]

c_logfc = np.log2(c_counts)
c_logfc = c_logfc.subtract(c_logfc[c_control], axis=0)
c_logfc = c_logfc.drop(c_control, axis=1)


# -
pal = {0: '#37454B', 1: '#F2C500'}

df_deseq = pd.read_csv('data/gdsc/crispr/deseq2/%s_logfold.tsv' % c_sample, sep='\t')

#
plot_df = pd.DataFrame({
    'log2fc': c_logfc.mean(1),
    'deseq_log2fc': df_deseq['log2FoldChange'],
    'gene': sgrna_lib['GENES']
}).dropna(subset=['gene'])
plot_df = plot_df.assign(cnv=list(cnv.loc[plot_df['gene'], manifest[c_sample]].apply(lambda v: int(v.split(',')[0]) if str(v) != 'nan' else np.nan)))
plot_df = plot_df.assign(nexp=[int(i in non_genes[manifest[c_sample]]) for i in plot_df['gene']])

#
plot_df = pd.DataFrame({
    'mean': c_counts.mean(1),
    'sd': c_counts.std(1),
    'cv': c_counts.std(1) / c_counts.mean(1),

    'sd_log2': np.log2(c_counts).std(1),
    'mean_log2': np.log2(c_counts).mean(1),

    'log2fc': c_logfc.mean(1),

    'deseq_mean': df_deseq['baseMean'],
    'deseq_log2fc': df_deseq['log2FoldChange']
}).drop('sgPOLR2K_1').sort_values('mean')
plot_df = plot_df.assign(gene=sgrna_lib.loc[plot_df.index, 'GENES'])
plot_df = plot_df.assign(essential=[int(i in essential) for i in plot_df['gene']])
plot_df = plot_df.dropna()
plot_df['cnv'] = list(cnv.loc[plot_df['gene'], manifest[c_sample]].apply(lambda v: int(v.split(',')[0]) if str(v) != 'nan' else np.nan))
plot_df['nexp'] = [int(i in non_genes[manifest[c_sample]]) for i in plot_df['gene']]

#
for x, y in [('mean', 'sd'), ('mean', 'cv'), ('mean_log2', 'sd_log2'), ('mean', 'log2fc'), ('deseq_mean', 'deseq_log2fc'), ('log2fc', 'deseq_log2fc')]:
    sns.set(rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5}, style='ticks', context='paper')
    g = sns.JointGrid(x, y, plot_df, space=0, ratio=8)

    def f(a, vertical=False, **kws):
        stripplot_args = {'data': kws['data'], 'size': 1, 'jitter': .2, 'palette': kws['palette'], 'alpha': .3}
        ax = sns.stripplot(x='essential', y=y, orient='v', **stripplot_args) if vertical else sns.stripplot(x=x, y='essential', orient='h', **stripplot_args)
        ax = sns.boxplot(x='essential', y=y, orient='v', **kws) if vertical else sns.boxplot(x=x, y='essential', orient='h', **kws)
        ax.set_ylabel('')
        ax.set_xlabel('')

    g.plot_marginals(f, palette=pal, data=plot_df, linewidth=.3, fliersize=1, notch=True, sym='')

    for i in pal:
        sns.regplot(
            x=x, y=y, data=plot_df.query('essential == %d' % i),
            truncate=True, fit_reg=False, color=pal[i], ax=g.ax_joint, label='Yes' if i else 'No',
            scatter_kws={'s': 2, 'alpha': .8}
        )

    g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
    g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)

    g.ax_joint.legend(loc=1, title='Essential')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/counts_%s_%s.png' % (x, y), bbox_inches='tight', dpi=600)
    plt.close('all')


