#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from crispy.utils import bin_cnv
from sklearn.metrics import roc_curve, auc


def ratios_histogram(x, data, outfile):
    f, axs = plt.subplots(2, 1, sharex=True)

    sns.distplot(data[x], color=bipal_dbgd[0], kde=False, axlabel='', ax=axs[0])
    sns.distplot(data[x], color=bipal_dbgd[0], kde=False, ax=axs[1])

    axs[0].axvline(1, lw=.3, ls='--', c=bipal_dbgd[1])
    axs[1].axvline(1, lw=.3, ls='--', c=bipal_dbgd[1])

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

    plt.title('Copy-number ratio histogram (non-expressed genes)')
    plt.gcf().set_size_inches(4, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def ratios_kmean(x, y, data, outfile):
    order = natsorted(set(data[y]))
    pal = sns.light_palette(bipal_dbgd[0], n_colors=len(order)).as_hex()

    sns.boxplot(x, y, data=data, orient='h', linewidth=.3, fliersize=1, order=order, palette=pal, notch=True)
    plt.axvline(0, lw=.1, ls='-', c=bipal_dbgd[0])

    plt.title('Copy-number ratio effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.xlabel('CRISPR-Cas9 fold-change (log2)')
    plt.ylabel('Copy-number ratio')
    plt.gcf().set_size_inches(4, 2)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def ratios_arocs(y_true, y_pred, data, outfile, exclude_labels=set()):
    order = natsorted(set(data[y_true]).difference(exclude_labels))

    pal = sns.light_palette(bipal_dbgd[0], n_colors=len(order) + 1).as_hex()[1:]

    ax = plt.gca()

    for c, thres in zip(pal, order):
        fpr, tpr, _ = roc_curve((data[y_true] == thres).astype(int), -data[y_pred])
        ax.plot(fpr, tpr, label='%s: AUC=%.2f' % (thres, auc(fpr, tpr)), lw=1.5, c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Copy-number ratio impact in CRISPR-Cas9\n(non-expressed genes)')
    legend = ax.legend(loc=4, title='Copy-number ratio', prop={'size': 10})
    legend.get_title().set_fontsize('10')

    plt.gcf().set_size_inches(3.5, 3.5)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def ratios_heatmap(x, y, z, data, outfile, z_bin='1'):
    plot_df = data.query("{} == '{}'".format(z, z_bin))

    counts = {(g, c) for g, c, v in plot_df.groupby([x, y])[z].count().reset_index().values if v == 1}
    plot_df = plot_df[[(g, c) not in counts for g, c in plot_df[[x, y]].values]]

    plot_df = plot_df.groupby([x, y])[z].count().reset_index()
    plot_df = pd.pivot_table(plot_df, index=x, columns=y, values=z)

    g = sns.heatmap(
        plot_df.loc[natsorted(plot_df.index, reverse=False)].T, cmap=sns.light_palette(bipal_dbgd[0], as_cmap=True), annot=True, fmt='g', square=True,
        linewidths=.3, cbar=False, annot_kws={'fontsize': 7}
    )
    plt.setp(g.get_yticklabels(), rotation=0)
    plt.ylabel('# chromosome copies')
    plt.xlabel('# gene copies')
    plt.title('Number of occurrences\n(non-expressed genes with copy-number ratio ~1)')
    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def ratios_heatmap_bias(x, y, z, data, outfile, z_bin='1'):
    plot_df = data.query("{} == '{}'".format('ratio_bin', z_bin))

    counts = {(g, c) for g, c, v in plot_df.groupby([x, y])[z].count().reset_index().values if v == 1}
    plot_df = plot_df[[(g, c) not in counts for g, c in plot_df[[x, y]].values]]

    plot_df = plot_df.groupby([x, y])[z].mean().reset_index()
    plot_df = pd.pivot_table(plot_df, index=x, columns=y, values=z)

    g = sns.heatmap(
        plot_df.loc[natsorted(plot_df.index, reverse=False)].T, cmap=sns.light_palette(bipal_dbgd[0], as_cmap=True, reverse=True), annot=True, fmt='.2f', square=True,
        linewidths=.3, cbar=False, annot_kws={'fontsize': 7}
    )
    plt.setp(g.get_yticklabels(), rotation=0)
    plt.ylabel('# chromosome copies')
    plt.xlabel('# gene copies')
    plt.title('Mean CRISPR-Cas9 fold-change\n(non-expressed genes with copy-number ratio ~1)')
    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


if __name__ == '__main__':
    # - Import
    # CRISPR library
    lib = pd.read_csv(mp.LIBRARY, index_col=0).groupby('gene')['chr'].first()

    # Non-expressed genes
    nexp = pd.read_csv(mp.NON_EXP, index_col=0)

    # Copy-number
    cnv = pd.read_csv(mp.CN_GENE.format('snp'), index_col=0)

    # Chromosome copies
    chrm = pd.read_csv(mp.CN_CHR.format('snp'), index_col=0)

    # Copy-number ratios
    cnv_ratios = pd.read_csv(mp.CN_GENE_RATIO.format('snp'), index_col=0)

    # CRISPR
    c_gdsc_fc = pd.read_csv(mp.CRISPR_GENE_FC, index_col=0)

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Assemble data-frame of non-expressed genes
    # Non-expressed genes CRISPR bias from copy-number
    df = pd.DataFrame({c: c_gdsc_fc[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples})

    # Concat copy-number information
    df = pd.concat([df[samples].unstack().rename('crispy'), cnv[samples].unstack().rename('cnv'), cnv_ratios[samples].unstack().rename('ratio')], axis=1).dropna()
    df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

    # Append chromossome copy-number information
    df = df.assign(chr=lib[df['gene']].values)
    df = df.assign(chr_cnv=chrm.unstack().loc[list(map(tuple, df[['sample', 'chr']].values))].values)

    # Bin copy-number profiles
    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cnv_bin=df['cnv'].apply(lambda v: bin_cnv(v, thresold=10)))
    df = df.assign(chr_cnv_bin=df['chr_cnv'].apply(lambda v: bin_cnv(v, thresold=6)))

    # - Plot: Copy-number ratio vs CRISPR bias
    ratios_histogram('ratio', df, 'reports/crispy/copynumber_ratio_histogram.png')
    ratios_kmean('crispy', 'ratio_bin', df, 'reports/crispy/copynumber_ratio_kmean_boxplot.png')
    ratios_arocs('ratio_bin', 'crispy', df, 'reports/crispy/copynumber_ratio_kmean_arocs.png')
    sns.set(style='white')
    ratios_heatmap('cnv_bin', 'chr_cnv_bin', 'ratio_bin', df, 'reports/crispy/copynumber_ratio_heatmap.png', z_bin='1')
    ratios_heatmap_bias('cnv_bin', 'chr_cnv_bin', 'crispy', df, 'reports/crispy/copynumber_ratio_heatmap_crispr.png')
