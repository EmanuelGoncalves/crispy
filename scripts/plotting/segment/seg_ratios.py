#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import pearsonr
from crispy.utils import bin_cnv
from scripts.plotting import bias_boxplot


def import_seg_crispr_bed(sample):
    return pd.read_csv('data/crispr_bed/{}.crispr.snp.bed'.format(sample), sep='\t')


def ratios_heatmap_bias(data, x='cn_bin', y='chrm_bin', z='fc', min_evetns=10):
    plot_df = data.copy()

    counts = {(g, c) for g, c, v in plot_df.groupby([x, y])[z].count().reset_index().values if v <= min_evetns}

    plot_df = plot_df[[(g, c) not in counts for g, c in plot_df[[x, y]].values]]
    plot_df = plot_df.groupby([x, y])[z].mean().reset_index()
    plot_df = pd.pivot_table(plot_df, index=x, columns=y, values=z)

    g = sns.heatmap(
        plot_df.loc[natsorted(plot_df.index, reverse=False)].T, cmap='RdGy_r', annot=True, fmt='.2f', square=True,
        linewidths=.3, cbar=False, annot_kws={'fontsize': 7}, center=0
    )
    plt.setp(g.get_yticklabels(), rotation=0)

    plt.ylabel('Chromosome copy-number')
    plt.xlabel('Segment absolute copy-number')

    plt.title('Mean CRISPR-Cas9 fold-change (non-expressed genes)')


def ratios_fc_boxplot(plot_df, x='ratio_bin', y='fc'):
    ax = bias_boxplot(plot_df, x=x, y=y)

    ax.axhline(0, ls='-', lw=.1, c=cy.PAL_DBGD[0], zorder=0)

    plt.title('Copy-number ratio effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.ylabel('Fold-change')
    plt.xlabel('Gene copy-number ratio')



def lm_fc(Y, X, n_splits=10, test_size=.5, ploidy=None, chrm=None):
    genes = set(Y.index).intersection(X.index)

    lib = cy.get_crispr_lib().groupby('gene')['chr'].first()[genes]

    gene_cvs = []
    for gene in genes:
        # Build data-frame
        y = Y.loc[gene, samples].dropna()
        x = X.loc[[gene], y.index].T

        if ploidy is not None:
            x = pd.concat([x, ploidy[samples]], axis=1)

        if chrm is not None:
            x = pd.concat([x, chrm.loc[lib[gene], samples]], axis=1)

        # Cross-validation
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        for i, (train_idx, test_idx) in enumerate(cv.split(y)):
            # Train model
            lm = LinearRegression().fit(x.iloc[train_idx], y.iloc[train_idx])

            # Evalutate model
            r2 = lm.score(x.iloc[test_idx], y.iloc[test_idx])

            # Store
            gene_cvs.append(dict(r2=r2, gene=gene))

    gene_cvs = pd.DataFrame(gene_cvs).groupby('gene').median()

    return gene_cvs



if __name__ == '__main__':
    # - Imports
    # Samples
    samples = mp.get_sample_list()

    # SNP6 + CRISPR BED files
    beds = {s: import_seg_crispr_bed(s) for s in samples}

    # - Aggregate per segment
    agg_fun = dict(fc=np.mean, cn=np.mean, chrm=np.mean, ploidy=np.mean, ratio=np.mean, len=np.mean)

    df = pd.concat([
        beds[s].groupby(['chr', 'start', 'end'])[list(agg_fun.keys())].agg(agg_fun).reset_index().assign(sample=s) for s in samples
    ])

    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cn_bin=df['cn'].apply(lambda v: bin_cnv(v, thresold=10)))
    df = df.assign(chrm_bin=df['chrm'].apply(lambda v: bin_cnv(v, thresold=6)))
    df = df.assign(ploidy_bin=df['ploidy'].apply(lambda v: bin_cnv(v, thresold=5)))

    df = df.assign(len_log10=np.log10(df['len']))

    # -
    pred_cnv = lm_fc(crispr.loc[genes, samples], cnv.loc[genes, samples])
    pred_cnv_covars = lm_fc(crispr.loc[genes, samples], cnv.loc[genes, samples], ploidy=ploidy, chrm=chrm)


    # Copy-number ratio bias heatmap
    ratios_heatmap_bias(df)

    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/seg_fc_heatmap_ploidy.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Copy-number ratio vs CRISPR bias boxplot
    ratios_fc_boxplot(df)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/seg_fc_boxplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    len_corr = {}
    for s in samples:
        df_ = df.query("sample == '{}'".format(s))
        r, p = pearsonr(df_['len_log10'], df_['fc'])
        len_corr[s] = r

    len_corr = pd.Series(len_corr).sort_values()

    sns.jointplot(df_['len_log10'], df_['cn'])
    # plt.gcf().set_size_inches(10, 3)
    plt.savefig('reports/seg_len_lmplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')
