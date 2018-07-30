#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy as cy
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.utils import bin_cnv
from scripts.plotting import bias_boxplot, bias_arocs


def ratios_histogram(plot_df):
    f, axs = plt.subplots(2, 1, sharex=True)

    c = cy.PAL_DBGD[0]

    sns.distplot(plot_df['ratio'], color=c, kde=False, axlabel='', ax=axs[0], hist_kws=dict(linewidth=0))
    sns.distplot(plot_df['ratio'], color=c, kde=False, ax=axs[1], hist_kws=dict(linewidth=0))

    axs[0].axvline(1, lw=.3, ls='--', c=c)
    axs[1].axvline(1, lw=.3, ls='--', c=c)

    axs[0].set_ylim(500000, 650000)  # outliers only
    axs[1].set_ylim(0, 200000)  # most of the data

    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].xaxis.tick_top()
    axs[0].tick_params(labeltop='off')
    axs[1].xaxis.tick_bottom()

    d = .01

    kwargs = dict(transform=axs[0].transAxes, color='gray', clip_on=False, lw=.3)
    axs[0].plot((-d, +d), (-d, +d), **kwargs)
    axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=axs[1].transAxes)
    axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axs[1].set_xlabel('Copy-number ratio (Gene / Chromosome)')

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)


def ratios_fc_boxplot(plot_df):
    ax = bias_boxplot(plot_df, x='ratio_bin', y='fc')

    ax.axhline(0, ls='-', lw=.1, c=cy.PAL_DBGD[0], zorder=0)

    plt.title('Copy-number ratio effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.ylabel('Fold-change')
    plt.xlabel('Gene copy-number ratio')


def ratios_heatmap_bias(data, min_evetns=10):
    x, y, z = 'cnv_bin', 'chr_cnv_bin', 'fc'

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
    plt.xlabel('Gene absolute copy-number')

    plt.title('Mean CRISPR-Cas9 fold-change (non-expressed genes)')


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

    # Copy-number ratios
    cnv_ratios = mp.get_copynumber_ratios()

    # CRISPR
    crispr = mp.get_crispr(mp.CRISPR_GENE_FC)

    # Pancancer core essential
    pancan_core = cy.get_adam_core_essential()

    # - Overlap samples
    samples = list(set(cnv).intersection(crispr).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Assemble data-frame of non-expressed genes
    # Non-expressed genes CRISPR bias from copy-number
    df = pd.DataFrame({c: crispr[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples})

    # Concat copy-number information
    df = pd.concat([df[samples].unstack().rename('fc'), cnv[samples].unstack().rename('cnv'), cnv_ratios[samples].unstack().rename('ratio')], axis=1).dropna()
    df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

    # Append chromossome copy-number information
    df = df.assign(chr=lib[df['gene']].values)
    df = df.assign(chr_cnv=chrm.unstack().loc[list(map(tuple, df[['sample', 'chr']].values))].values)
    df = df.assign(ploidy=ploidy[df['sample']].values)

    # Bin copy-number profiles
    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cnv_bin=df['cnv'].apply(lambda v: bin_cnv(v, thresold=10)))
    df = df.assign(chr_cnv_bin=df['chr_cnv'].apply(lambda v: bin_cnv(v, thresold=6)))
    df = df.assign(ploidy_bin=df['ploidy'].apply(lambda v: bin_cnv(v, thresold=5)))

    # Filter
    df = df[~df['gene'].isin(pancan_core)]

    # - Plot
    # Copy-number ratio histogram
    ratios_histogram(df)

    plt.title('Copy-number ratio histogram (non-expressed genes)')
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('reports/ratio_histogram.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Copy-number ratio vs CRISPR bias boxplot
    ratios_fc_boxplot(df)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/ratio_fc_boxplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Copy-number ratio bias heatmap
    ratios_heatmap_bias(df)

    plt.title('Mean CRISPR-Cas9 fold-change\n(non-expressed genes)')

    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/ratio_fc_heatmap.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Overall copy-number ratio bias AUCs
    ax, aucs = bias_arocs(df, groupby='ratio_bin')
    ax.set_title('Copy-number ratio effect on CRISPR-Cas9\n(non-expressed genes)')
    ax.legend(loc=4, title='Copy-number ratio', prop={'size': 6}, frameon=False).get_title().set_fontsize(6)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/ratios_aucs.png', bbox_inches='tight', dpi=600)
    plt.close('all')
