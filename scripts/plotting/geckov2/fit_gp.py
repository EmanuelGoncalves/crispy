#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import pearsonr
from crispy.utils import bin_cnv
from scripts.plotting import bias_boxplot
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.gaussian_process import GaussianProcessRegressor
from scripts.plotting.qc_aucs import aucs_curves, aucs_scatter
from scripts.plotting import bias_boxplot, bias_arocs, aucs_boxplot
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF


GECKOV2_SGRNA_MAP = 'data/geckov2/Achilles_v3.3.8_sgRNA_mappings.txt'


def import_seg_crispr_bed(sample):
    return pd.read_csv(f'data/geckov2/crispr_bed/{sample}.crispr.snp.bed', sep='\t')


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


def lm_fc(y, X, n_splits=10, test_size=.3):
    # Cross-validation
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Linear regression
    r2s = []
    for i, (train_idx, test_idx) in enumerate(cv.split(y)):
        # Train model
        kernel = ConstantKernel() * RBF(length_scale_bounds=(1e-5, 10)) + WhiteKernel()

        gpr = GaussianProcessRegressor(kernel, alpha=1e-10, n_restarts_optimizer=3)

        gpr = gpr.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Evalutate model
        r2 = gpr.score(X.iloc[test_idx], y.iloc[test_idx])

        # Store
        r2s.append(dict(r2=r2, sample=y.name))

    r2s = pd.DataFrame(r2s)

    return r2s


def gp_fit(y, X):
    # Train model
    kernel = ConstantKernel() * RBF(length_scale_bounds=(1e-5, 10)) + WhiteKernel()

    gpr = GaussianProcessRegressor(kernel, alpha=1e-10, n_restarts_optimizer=3)

    gpr = gpr.fit(X, y)

    gp_mean = pd.Series(gpr.predict(X), index=X.index).rename('gp_mean')

    return gp_mean


def import_lib():
    lib = pd.read_csv(GECKOV2_SGRNA_MAP, sep='\t').dropna(subset=['Chromosome', 'Pos', 'Strand'])

    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])

    sgrna_maps = lib['sgrna'].value_counts()
    lib = lib[lib['sgrna'].isin(sgrna_maps[sgrna_maps == 1].index)]

    lib = lib.rename(columns={'Chromosome': '#chr', 'Pos': 'start'})

    lib['start'] = lib['start'].astype(int)
    lib['end'] = lib['start'] + lib['sgrna'].apply(len) + lib['PAM'].apply(len)

    lib = lib.set_index('sgrna')

    lib['id'] = lib.index

    return lib


def aucs_scatter(x, y, data):
    g = sns.scatterplot(x, y, data=data)

    x0, x1 = g.get_xlim()
    y0, y1 = g.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.plot(lims, lims, ':k')

    plt.xlabel(f'{x.capitalize()} fold-changes AURCs')
    plt.ylabel(f'{y.capitalize()} fold-changes AURCs')


def bias_copynumber_aucs(plot_df, x_var):
    ax, aucs = bias_arocs(plot_df, x_rank=x_var, groupby='cn_bin')

    ax.set_title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')
    ax.legend(loc=4, title='Copy-number', prop={'size': 6}, frameon=False).get_title().set_fontsize(6)


def calculate_bias_by(plot_df, per, x_rank, groupby):
    aucs = {i: bias_arocs(df, to_plot=False, x_rank=x_rank, groupby=groupby)[1] for i, df in plot_df.groupby(per)}

    aucs = pd.DataFrame(aucs).unstack().dropna().rename('auc').reset_index()

    for i, n in enumerate(per):
        aucs = aucs.rename(columns={'level_{}'.format(i): n})

    aucs = aucs.rename(columns={'level_{}'.format(len(per)): 'cnv'})

    return aucs


if __name__ == '__main__':
    # - Imports
    # Gene-sets
    essential = pd.Series(list(cy.get_essential_genes())).rename('essential')
    nessential = pd.Series(list(cy.get_non_essential_genes())).rename('non-essential')

    # Samples
    samples = list(pd.read_csv('data/geckov2/Achilles_v3.3.8.samplesheet.txt', sep='\t', index_col=1)['name'])
    samples = [s for s in samples if os.path.exists(f'data/geckov2/crispr_bed/{s}.crispr.snp.bed')]

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

    df = df.set_index(['sample', 'chr', 'start', 'end'])

    # - Copy-number ratio bias heatmap
    ratios_heatmap_bias(df)
    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/geckov2_fc_heatmap.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    x_features = [('all', ['cn', 'chrm', 'ratio', 'len_log2']), ('cn', ['cn']), ('ratio', ['ratio'])]

    lm_r2s = pd.concat([
        lm_fc(
            df.query("sample == @c")['fc'].rename(c), df.query("sample == @c")[f]
        ).assign(type=n) for n, f in x_features for c in samples
    ]).reset_index(drop=True)

    # -
    plot_df = lm_r2s.groupby(['sample', 'type']).mean().reset_index()
    plot_df = pd.pivot_table(plot_df, index='sample', columns='type', values='r2')

    g = sns.scatterplot('ratio', 'cn', data=plot_df)

    x0, x1 = g.get_xlim()
    y0, y1 = g.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.plot(lims, lims, ':k')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/geckov2_r2_scatter.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    features = ['cn', 'chrm', 'ratio', 'len_log2']

    gp_mean = pd.concat([gp_fit(df.query("sample == @c")['fc'].rename(c), df.query("sample == @c")[features]) for c in samples])
    df = pd.concat([df, gp_mean], axis=1)

    # -
    bed_gp = {c: beds[c].assign(
        gp_mean=df.loc[[(c, chrm, start, end) for chrm, start, end in beds[c][['chr', 'start', 'end']].values], 'gp_mean'].values
    ) for c in beds}

    # -
    lib = import_lib()

    bed_gene = pd.concat([
        bed_gp[c]\
            .assign(corrected=bed_gp[c].eval('fc - gp_mean'))\
            .groupby(lib.loc[bed_gp[c]['sgrna_id']]['Guide'].apply(lambda v: v.split('_')[1]).values)['corrected']\
            .mean()\
            .rename(c) for c in bed_gp
    ], axis=1)
    bed_gene.columns.name = 'Crispy'

    # - Calculate AUCs of essential/non-essential genes in original/corrected fold-changes
    ceres_gfc = pd.read_csv('data/geckov2/Achilles_v3.3.8..ceres.gene.effects.csv', index_col=0)[bed_gene.columns]
    ceres_gfc.columns.name = 'CERES'

    aucs = []
    for df in [ceres_gfc, bed_gene]:
        for gset in [essential, nessential]:
            #
            df_aucs = aucs_curves(df[samples], gset)

            plt.gcf().set_size_inches(3, 3)
            plt.savefig(f'reports/geckov2_qc_aucs_{df.columns.name}_{gset.name}.png', bbox_inches='tight', dpi=600)
            plt.close('all')

            aucs.append(pd.DataFrame(dict(aurc=df_aucs['auc'], type=df.columns.name, geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})

    # - Scatter comparison
    for gset in [essential, nessential]:
        plot_df = pd.pivot_table(
            aucs.query("geneset == '{}'".format(gset.name)),
            index='sample', columns='type', values='aurc'
        )

        aucs_scatter('CERES', 'Crispy', plot_df)

        plt.title('AURCs {} genes'.format(gset.name))

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/geckov2_qc_aucs_{}.png'.format(gset.name), bbox_inches='tight', dpi=600)
        plt.close('all')

    # - Overall copy-number bias AUCs
    df_cn = []
    for c in beds:
        df_ = bed_gp[c].assign(gene=lib.loc[bed_gp[c]['sgrna_id']]['Guide'].apply(lambda v: v.split('_')[1]).values)

        df_ = df_.assign(crispy=bed_gene.loc[df_['gene'], c].values)

        df_ = df_.groupby(['gene'])[['crispy', 'cn']].mean()
        df_ = df_.assign(cn_bin=df_['cn'].apply(lambda v: bin_cnv(v, thresold=10)))

        df_ = df_.assign(ceres=ceres_gfc.loc[df_.index, c].values)

        df_ = df_.assign(sample=c)

        df_cn.append(df_.dropna())

    df_cn = pd.concat(df_cn)

    #
    bias_copynumber_aucs(df_cn, x_var='crispy')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/geckov2_bias_copynumber_aucs_crispy.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    bias_copynumber_aucs(df_cn, x_var='ceres')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/geckov2_bias_copynumber_aucs_ceres.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    aucs_samples_crispy = calculate_bias_by(df_cn, per='sample', x_rank='crispy', groupby='cn_bin')
    aucs_samples_ceres = calculate_bias_by(df_cn, per='sample', x_rank='ceres', groupby='cn_bin')

    plot_df = pd.concat([
        aucs_samples_crispy.set_index(['s', 'a'])['auc'].rename('crispy'),
        aucs_samples_ceres.set_index(['s', 'a'])['auc'].rename('ceres')
    ], axis=1)
    plot_df = plot_df.reset_index()

    order = ['4', '5', '6', '7', '8', '9', '10+']

    g = sns.jointplot('crispy', 'ceres', data=plot_df[plot_df['a'].isin(order)])

    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/geckov2_cn_aucs_scatter.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    def calculate_bias_aupr_by(df_cn, x_rank):
        aucs = {}

        for s in set(df_cn['sample']):
            df_s = df_cn.query('sample == @s')

            # p, r, _ = precision_recall_curve(df_s.index.isin(essential).astype(int), -df_s[x_rank])
            # aucs[s] = auc(r, p)

            aucs[s] = average_precision_score(df_s.index.isin(essential).astype(int), -df_s[x_rank])

        return pd.Series(aucs)


    auprs_samples_crispy = calculate_bias_aupr_by(df_cn, x_rank='crispy')
    auprs_samples_ceres = calculate_bias_aupr_by(df_cn, x_rank='ceres')

    plot_df = pd.concat([
        auprs_samples_crispy.rename('crispy'),
        auprs_samples_ceres.rename('ceres')
    ], axis=1)

    g = sns.jointplot('crispy', 'ceres', data=plot_df)

    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/geckov2_cn_auprs_scatter.png', bbox_inches='tight', dpi=600)
    plt.close('all')
