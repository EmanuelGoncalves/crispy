#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy as cy
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.utils import bin_cnv
from scripts.plotting import bias_boxplot, bias_arocs, aucs_boxplot


EXCLUDED_CNV_BINS = ['0', '1']


def bias_crispr_fc_boxplot(plot_df):
    ax = bias_boxplot(plot_df)

    ax.axhline(0., ls='-', lw=.1, c=cy.PAL_DBGD[0], zorder=0)

    ax.set_title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')
    ax.set_xlabel('Gene absolute copy-number')
    ax.set_ylabel('Fold-change')


def bias_copynumber_aucs(plot_df):
    ax, aucs = bias_arocs(plot_df)

    ax.set_title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')
    ax.legend(loc=4, title='Copy-number', prop={'size': 6}, frameon=False).get_title().set_fontsize(6)


def calculate_bias_by(plot_df, groupby):
    aucs = {i: bias_arocs(df, to_plot=False)[1] for i, df in plot_df.groupby(groupby)}

    aucs = pd.DataFrame(aucs).unstack().dropna().rename('auc').reset_index()

    for i, n in enumerate(groupby):
        aucs = aucs.rename(columns={'level_{}'.format(i): n})

    aucs = aucs.rename(columns={'level_{}'.format(len(groupby)): 'cnv'})

    return aucs


if __name__ == '__main__':
    # - Import
    # CRISPR library
    lib = cy.get_crispr_lib().groupby('gene')['chr'].first()

    # Non-expressed genes
    nexp = mp.get_non_exp()

    # Copy-number
    cnv = mp.get_copynumber()

    # Ploidy
    ploidy = mp.get_ploidy()

    # Chromosome copies
    chrm = mp.get_chr_copies()

    # CRISPR
    crispr = mp.get_crispr(mp.CRISPR_GENE_FC)
    crispr.columns.name = 'original'

    # Corrected CRISPR
    crispy = mp.get_crispr(mp.CRISPR_GENE_CORRECTED_FC)
    crispy.columns.name = 'corrected'

    # - Overlap samples
    samples = list(set(cnv).intersection(crispr).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Analyse original and corrected fold-changes
    for c_df in [crispr, crispy]:
        # - Essential genes AURCs
        essential_aurc = pd.read_csv('data/qc_aucs.csv')
        essential_aurc = essential_aurc.query("type == '{}'".format(c_df.columns.name))
        essential_aurc = essential_aurc.query("geneset == 'essential'")['aurc'].mean()

        # - Build data-frame
        df = pd.concat([
            pd.DataFrame({c: c_df[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples}).unstack().rename('fc'),
            cnv[samples].applymap(lambda v: bin_cnv(v, 10)).unstack().rename('cnv')
        ], axis=1).dropna()

        df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
        # df = df[~df['cnv'].isin(EXCLUDED_CNV_BINS)]
        df = df.assign(chr=lib.loc[df['gene']].values)

        # - Calculate AUC bias curves
        aucs_samples = calculate_bias_by(df, groupby=['sample'])
        aucs_samples = aucs_samples.assign(ploidy=ploidy[aucs_samples['sample']].apply(lambda v: bin_cnv(v, 5)).values)

        aucs_chr = calculate_bias_by(df, groupby=['sample', 'chr'])
        aucs_chr = aucs_chr.assign(ploidy=ploidy[aucs_chr['sample']].apply(lambda v: bin_cnv(v, 5)).values)
        aucs_chr = aucs_chr.assign(chr_copies=[bin_cnv(chrm.loc[c, s], 6) for s, c in aucs_chr[['sample', 'chr']].values])

        # - Plot
        # Copy-number bias in CRISPR fold-changes
        bias_crispr_fc_boxplot(df)

        plt.gcf().set_size_inches(2.5, 3)
        plt.savefig('reports/{}_bias_boxplot_fc.png'.format(c_df.columns.name), bbox_inches='tight', dpi=600)
        plt.close('all')

        # Overall copy-number bias AUCs
        bias_copynumber_aucs(df)

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/{}_bias_copynumber_aucs.png'.format(c_df.columns.name), bbox_inches='tight', dpi=600)
        plt.close('all')

        # Per sample copy-number bias boxplots
        aucs_boxplot(aucs_samples, x='cnv', y='auc', essential_line=essential_aurc)

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/{}_bias_copynumber_per_sample.png'.format(c_df.columns.name), bbox_inches='tight', dpi=600)
        plt.close('all')

        # Per sample copy-number bias boxplots by ploidy
        aucs_boxplot(aucs_samples, x='cnv', y='auc', hue='ploidy', essential_line=essential_aurc)

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/{}_bias_copynumber_per_sample_ploidy.png'.format(c_df.columns.name), bbox_inches='tight', dpi=600)
        plt.close('all')

        # Per chromosome copy-number bias boxplots
        ax = aucs_boxplot(aucs_chr, x='cnv', y='auc', essential_line=essential_aurc)

        ax.set_ylabel('Copy-number AURC (per chromossome)')

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/{}_bias_copynumber_per_chromosome.png'.format(c_df.columns.name), bbox_inches='tight', dpi=600)
        plt.close('all')

        # Per chromosome copy-number bias boxplots by ploidy
        ax = aucs_boxplot(aucs_chr, x='cnv', y='auc', hue='ploidy', essential_line=essential_aurc)

        ax.set_ylabel('Copy-number AURC (per chromossome)')

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/{}_bias_copynumber_per_chromosome_ploidy.png'.format(c_df.columns.name), bbox_inches='tight', dpi=600)
        plt.close('all')
