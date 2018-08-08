#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import scripts as mp
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import pearsonr
from crispy.utils import bin_cnv
from scripts.plotting import bias_boxplot
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF


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


def lm_fc(y, X, n_splits=10, test_size=.3, normalize=False):
    # Cross-validation
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Linear regression
    r2s = []
    for i, (train_idx, test_idx) in enumerate(cv.split(y)):
        # Train model
        # lm = LinearRegression(normalize=normalize).fit(X.iloc[train_idx], y.iloc[train_idx])
        kernel = ConstantKernel() * RBF(length_scale_bounds=(1e-5, 10)) + WhiteKernel()

        gpr = GaussianProcessRegressor(kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=3)

        gpr = gpr.fit(X.iloc[test_idx], y.iloc[test_idx])

        # Evalutate model
        r2 = gpr.score(X.iloc[test_idx], y.iloc[test_idx])

        # Store
        r2s.append(dict(r2=r2, sample=y.name))

    r2s = pd.DataFrame(r2s)

    return r2s


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

    df = df.assign(len_log2=np.log2(df['len']))

    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cn_bin=df['cn'].apply(lambda v: bin_cnv(v, thresold=10)))
    df = df.assign(chrm_bin=df['chrm'].apply(lambda v: bin_cnv(v, thresold=6)))
    df = df.assign(ploidy_bin=df['ploidy'].apply(lambda v: bin_cnv(v, thresold=5)))

    # - Copy-number ratio bias heatmap
    ratios_heatmap_bias(df)
    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/seg_fc_heatmap.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    x_features = [('all', ['cn', 'chrm', 'ratio', 'len_log2']), ('cn', ['cn'])]

    lm_r2s = pd.concat([
        lm_fc(
            df.query("sample == @c")['fc'].rename(c), df.query("sample == @c")[f]
        ).assign(type=n) for n, f in x_features for c in samples
    ]).reset_index(drop=True)

    # -
    plot_df = lm_r2s.groupby(['sample', 'type']).median().reset_index()
    plot_df = pd.pivot_table(plot_df, index='sample', columns='type', values='r2')

    g = sns.scatterplot('all', 'cn', data=plot_df)

    x0, x1 = g.get_xlim()
    y0, y1 = g.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.plot(lims, lims, ':k')

    plt.show()