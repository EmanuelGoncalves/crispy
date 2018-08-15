#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.utils import bin_cnv
from scripts.plotting import bias_arocs
from plotting import plot_chromosome_from_bed
from sklearn.model_selection import ShuffleSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF


def import_seg_crispr_bed(sample):
    return pd.read_csv('data/crispr_bed/{}.crispr.snp.bed'.format(sample), sep='\t')


def get_gpr(length_scale_bounds=(1e-5, 10), alpha=1e-10, normalize_y=False, n_restarts_optimizer=3):
    # Kernel
    kernel = ConstantKernel() * RBF(length_scale_bounds=length_scale_bounds) + WhiteKernel()

    # Gaussian process
    gpr = GaussianProcessRegressor(kernel, alpha=alpha, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer)

    return gpr


def gpr_fc(y, X, n_splits=10, test_size=.3):
    # Cross-validation
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Linear regression
    r2s = []
    for i, (train_idx, test_idx) in enumerate(cv.split(y)):
        # Initialise Gaussian Process Regressor
        gpr = get_gpr()

        # Train model
        gpr = gpr.fit(X.iloc[test_idx], y.iloc[test_idx])

        # Evalutate model
        r2 = gpr.score(X.iloc[test_idx], y.iloc[test_idx])

        # Store
        r2s.append(dict(r2=r2, sample=y.name))

    r2s = pd.DataFrame(r2s)

    return r2s


def fit_gp(df_s):
    gp = get_gpr().fit(df_s[x_features['all']], df_s['fc'])

    gp_pred = pd.Series(gp.predict(df_s[x_features['all']]), index=df_s.index)

    return gp_pred


def calculate_bias_by(bed_gp, x_rank, groupby):
    agg_fun = {x_rank: 'mean', groupby: lambda v: bin_cnv(np.median(v), 10)}

    aucs = []
    for s in bed_gp:
        print(f'[INFO] Bias analysis: {s}')

        df_s_auc = bed_gp[s].groupby('gene')[[x_rank, groupby]].agg(agg_fun)
        df_s_auc = bias_arocs(df_s_auc, x_rank=x_rank, groupby=groupby, to_plot=False)[1]
        df_s_auc = pd.Series(df_s_auc).rename('auc').reset_index().rename(columns=dict(index=groupby)).assign(sample=s)

        aucs.append(df_s_auc)

    aucs = pd.concat(aucs).reset_index(drop=True)

    return aucs


def get_mobem(drop_factors=True):
    index = 'COSMIC ID'

    ss = pd.read_csv('/Users/eg14/Projects/dtrace/data/meta/samplesheet.csv').dropna(subset=[index]).set_index(index)

    mobem = pd.read_csv('/Users/eg14/Projects/dtrace/data/PANCAN_mobem.csv', index_col=0)
    mobem = mobem.set_index(ss.loc[mobem.index, 'Cell Line Name'], drop=True)
    mobem = mobem.T

    if drop_factors:
        mobem = mobem.drop(['TISSUE_FACTOR', 'MSI_FACTOR', 'MEDIA_FACTOR'])

    mobem = mobem.astype(int)

    return mobem


if __name__ == '__main__':
    # - Imports
    # CRISPR library
    lib = cy.get_crispr_lib()

    # Samples
    samples = mp.get_sample_list()

    # SNP6 + CRISPR BED files
    beds = {s: import_seg_crispr_bed(s) for s in samples}

    #
    mobem = get_mobem()
    wes = pd.read_csv('/Users/eg14/Projects/prioritize_crispr/data/WES_variants.csv')
    ss = pd.read_csv('/Users/eg14/Projects/dtrace/data/meta/samplesheet.csv', index_col=0)

    # Ploidy
    ploidy = pd.Series({s: beds[s]['ploidy'].mean() for s in beds})

    # Import CRISPRcleanR corrected fold-changes
    ccleanr_gene = pd.read_csv('/Users/eg14/Projects/prioritize_crispr/data/quantile_norm_CRISPRcleaned_logFCs.tsv', sep='\t', index_col=0)

    # - Aggregate per segment
    agg_fun = dict(fc=np.mean, cn=np.mean, chrm=np.mean, ploidy=np.mean, ratio=np.mean, len=np.mean, sgrna_id='count')

    df = pd.concat([
        beds[s].groupby(['chr', 'start', 'end'])[list(agg_fun.keys())].agg(agg_fun).reset_index().assign(sample=s) for s in samples
    ])

    df = df.assign(len_log2=np.log2(df['len']))

    df = df.set_index(['sample', 'chr', 'start', 'end'])

    #
    klm = df[~df['chr'].isin(['chrY', 'chrX'])].groupby('sample')['ratio'].mean()
    klm = pd.concat([klm, ss['Cancer Type']], axis=1).dropna()
    klm = klm.assign(type=klm['Cancer Type'].isin(['Breast Carcinoma']).astype(int))

    sns.boxplot('type', 'ratio', data=klm)
    plt.show()

    # - Sample segment fit
    x_features = dict(
        cn=['cn'], ratio=['ratio'], all=['cn', 'chrm', 'ratio', 'len_log2']
    )

    gpr_r2s = pd.concat([
        gpr_fc(
            df.query("sample == @c")['fc'].rename(c), df.query("sample == @c")[x_features[n]]
        ).assign(type=n) for n in x_features for c in samples
    ]).reset_index(drop=True)

    # - Plot compare predictive power of different set of features
    plot_df = gpr_r2s.groupby(['sample', 'type']).mean().reset_index()
    plot_df = pd.pivot_table(plot_df, index='sample', columns='type', values='r2')

    for xx, yy, x_label in [('ratio', 'cn', 'Copy-number ratio'), ('all', 'cn', 'All features')]:
        g = sns.scatterplot(xx, yy, data=plot_df, color=cy.PAL_DBGD[0], lw=.1)

        (x0, x1), (y0, y1) = g.get_xlim(), g.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.plot(lims, lims, color=cy.PAL_DBGD[0], lw=.3, zorder=0)

        plt.ylabel('Copy-number\n(CV mean R-squared)')
        plt.xlabel(f'{x_label}\n(CV mean R-squared)')

        plt.title('Predicting copy-number effect\non CRISPR-Cas9')

        plt.gcf().set_size_inches(2, 2)
        plt.savefig(f'reports/gp_seg_r2_{xx}_{yy}.png', bbox_inches='tight', dpi=600)
        plt.close('all')

    # - Fit GPR mean per segment
    df = pd.concat([df, pd.concat([fit_gp(df.query("sample == @s")).rename('gp_mean') for s in samples])], axis=1)

    # - Add segment estimated bias to original bed files
    bed_gp = {}
    for s in samples:
        guides_gpmean = df.loc[[(s, chrm, start, end) for chrm, start, end in beds[s][['chr', 'start', 'end']].values], 'gp_mean'].values
        bed_s = beds[s].assign(gp_mean=guides_gpmean)

        bed_s = bed_s.assign(gene=lib.loc[bed_s['sgrna_id'], 'GENES'].values)

        bed_s = bed_s.assign(fc_corrected=bed_s.eval('fc - gp_mean'))
        bed_s = bed_s.assign(fc_ccleanr=ccleanr_gene.loc[bed_s['gene'], s].values)

        bed_s = bed_s.assign(cn_bin=bed_s['cn'].apply(lambda v: bin_cnv(v, thresold=10)))
        bed_s = bed_s.assign(ratio_bin=bed_s['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
        bed_s = bed_s.assign(chrm_bin=bed_s['chrm'].apply(lambda v: bin_cnv(v, thresold=6)))
        bed_s = bed_s.assign(ploidy_bin=bed_s['ploidy'].apply(lambda v: bin_cnv(v, thresold=5)))

        bed_s.to_csv(f'data/crispr_bed/{s}.crispy.bed', sep='\t')

        bed_gp[s] = bed_s

    for s in samples:
        bed_gp[s].to_csv(f'data/crispr_bed/{s}.crispy.bed', sep='\t', index=False)

    # - Calculate copy-number bias
    aucs_crispy = calculate_bias_by(bed_gp, x_rank='fc_corrected', groupby='cn')
    aucs_ccleanr = calculate_bias_by(bed_gp, x_rank='fc_ccleanr', groupby='cn')

    # Plot
    plot_df = pd.concat([
        aucs_crispy.groupby('sample')['auc'].mean().rename('crispy'),
        aucs_ccleanr.groupby('sample')['auc'].mean().rename('ccleanr')
    ], axis=1).reset_index()
    plot_df = plot_df.assign(diff=plot_df.eval('crispy - ccleanr'))

    g = sns.jointplot(
        'crispy', 'ccleanr', data=plot_df, kind='scatter', space=0, color=cy.PAL_DBGD[0],
        marginal_kws=dict(kde=False), joint_kws=dict(edgecolor='w', lw=.3, s=10, alpha=.6)
    )

    (x0, x1), (y0, y1) = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, color=cy.PAL_DBGD[0], lw=.3, zorder=0)

    g.ax_joint.axvline(plot_df['crispy'].mean(), lw=.3, c=cy.PAL_DBGD[1], zorder=0)
    g.ax_joint.axhline(plot_df['ccleanr'].mean(), lw=.3, c=cy.PAL_DBGD[1], zorder=0)

    g.set_axis_labels('Crispy (AUPR)', 'CRISPRcleanR (AUPR)')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/gp_auprs_scatter.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    s = 'HT-29'

    bed_gp[s].query("cn_bin == '8'").sort_values('fc_ccleanr')

    plot_chromosome_from_bed(s, 'chr8', genes_highlight=[])
    plt.gcf().set_size_inches(5, 3)
    plt.savefig('reports/gp_chromosome_plot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    xx, yy = ['cn', 'chrm', 'ratio', 'len_log2'], 'fc'

    plot_df = df.query('sample == @s').sort_values('cn')

    gp = get_gpr().fit(plot_df[xx], plot_df[yy])

    y_gpr, y_std = gp.predict(plot_df[xx], return_std=True)

    plt.scatter(plot_df['cn'], plot_df[yy], c=cy.PAL_DBGD[0], label='data', zorder=2)
    plt.plot(plot_df['cn'], y_gpr, color=cy.PAL_DBGD[2], lw=1., label='GPR')
    plt.fill_between(plot_df['cn'], y_gpr - y_std, y_gpr + y_std, color=cy.PAL_DBGD[2], alpha=0.5)

    plt.legend(frameon=False, loc=4)

    plt.show()
