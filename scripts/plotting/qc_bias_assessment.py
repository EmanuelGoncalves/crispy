#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from natsort import natsorted
from sklearn.metrics import auc
from crispy.utils import bin_cnv
from crispy.benchmark_plot import plot_cumsum_auc


def tissue_coverage(x, y, data, outfile):
    df = data.assign(ypos=np.arange(data.shape[0]))

    plt.barh(df['ypos'], df[x], .8, color=bipal_dbgd[0], align='center')

    for xx, yy in df[['ypos', x]].values:
        plt.text(yy - .25, xx, str(yy), color='white', ha='right', va='center', fontsize=6)

    plt.yticks(df['ypos'])
    plt.yticks(df['ypos'], df[y])

    plt.xlabel('# cell lines')
    plt.title('CRISPR-Cas9 screen\n(%d cell lines)' % df['Counts'].sum())

    plt.gcf().set_size_inches(2, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def auc_curves(df, geneset, outfile):
    # Palette
    pal = sns.light_palette(bipal_dbgd[0], n_colors=df.shape[1]).as_hex()

    # AUCs fold-changes
    ax, stats_ess = plot_cumsum_auc(df, geneset, plot_mean=False, legend=False, palette=pal)
    plt.title('Mean AURC = {:.2f}'.format(pd.Series(stats_ess['auc']).mean()))
    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')

    return stats_ess


def aucs_scatter(x, y, data, outfile, title):
    ax = plt.gca()

    ax.scatter(data[x], data[y], c=bipal_dbgd[0], s=3, marker='x', lw=0.5)
    sns.rugplot(data[x], height=.02, axis='x', c=bipal_dbgd[0], lw=.3, ax=ax)
    sns.rugplot(data[y], height=.02, axis='y', c=bipal_dbgd[0], lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('{} fold-changes AURCs'.format(x))
    ax.set_ylabel('{} fold-changes AURCs'.format(y))
    ax.set_title(title)

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def copy_number_bias_aucs(df, outfile=None, thres_label='cnv', rank_label='fc', min_events=5, legend_size=6, title=None, legend_title=None):
    aucs = {}

    ax = plt.gca() if outfile is not None else None

    thresholds = natsorted(set(df[thres_label]))
    thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(thresholds)).as_hex()

    for i, t in enumerate(thresholds):
        index_set = set(df[df[thres_label] == t].index)

        if len(index_set) >= min_events:
            # Build data-frame
            x = df[rank_label].sort_values().dropna()

            # Observed cumsum
            y = x.index.isin(index_set)
            y = np.cumsum(y) / sum(y)

            # Rank fold-changes
            x = st.rankdata(x) / x.shape[0]

            # Calculate AUC
            f_auc = auc(x, y)
            aucs[t] = f_auc

            # Plot
            if outfile is not None:
                ax.plot(x, y, label='{}: AURC={:.2f}'.format(t, f_auc), lw=1., c=thresholds_color[i])

    if outfile is not None:
        # Random
        ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

        # Limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Labels
        ax.set_xlabel('Ranked')
        ax.set_ylabel('Recall')

        ax.legend(loc=4)

        ax.set_title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)' if title is None else title)
        legend = ax.legend(loc=4, title='Copy-number' if legend_title is None else legend_title, prop={'size': legend_size})
        legend.get_title().set_fontsize('{}'.format(legend_size))

        # plt.show()
        plt.gcf().set_size_inches(3, 3)
        plt.savefig(outfile, bbox_inches='tight', dpi=600)
        plt.close('all')

    return ax, aucs


def copy_number_bias_aucs_per_sample(x, y, hue, data, outfile):
    thresholds = natsorted(set(data[x]))
    thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(thresholds)).as_hex()

    # Plot per sample
    sns.boxplot(x, y, data=data, sym='', order=thresholds, palette=thresholds_color, linewidth=.3)
    sns.stripplot(x, y, data=data, order=thresholds, palette=thresholds_color, edgecolor='white', linewidth=.1, size=3, jitter=.4)

    plt.ylim(0, 1)
    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')
    plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Plot per sample hue by ploidy
    hue_thresholds = natsorted(set(data[hue]))
    hue_thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(hue_thresholds)).as_hex()

    sns.boxplot(x, y, hue, data=data, sym='', order=thresholds, hue_order=hue_thresholds, palette=hue_thresholds_color, linewidth=.3)
    sns.stripplot(x, y, hue, data=data, order=thresholds, hue_order=hue_thresholds, palette=hue_thresholds_color, edgecolor='white', linewidth=.1, size=1, jitter=.2, split=True)

    plt.ylim(0, 1)
    handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(hue_thresholds_color, hue_thresholds))]
    plt.legend(handles=handles, title='Ploidy', prop={'size': 6}).get_title().set_fontsize('6')
    plt.title('Cell ploidy effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')
    plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
    plt.gcf().set_size_inches(3, 3)

    outfile_dir, outfile_exp = os.path.splitext(outfile)
    plt.savefig('{}_ploidy.{}'.format(outfile_dir, outfile_exp), bbox_inches='tight', dpi=600)
    plt.close('all')


def copy_number_bias_aucs_per_chrm(x, y, hue, data, outfile):
    # Copy-number bias per chromosome
    order = natsorted(set(data[x]))

    hue_order = natsorted(set(data[hue]))
    hue_color = sns.light_palette(bipal_dbgd[0], n_colors=len(hue_order)).as_hex()

    sns.boxplot(x, y, hue, data=data, hue_order=hue_order, order=order, sym='', palette=hue_color, linewidth=.3)
    sns.stripplot(x, y, hue, data=data, hue_order=hue_order, order=order, palette=hue_color, edgecolor='white', size=.5, jitter=.2, split=True)

    handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(hue_color, hue_order))]
    plt.legend(handles=handles, title='Chr. copies', prop={'size': 5}).get_title().set_fontsize('5')

    plt.ylim(0, 1)
    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per chromossome)')
    plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def arocs_scatter(x, y, data, outfile):
    g = sns.jointplot(data[x], data[y], color=bipal_dbgd[0], space=0, kind='scatter', joint_kws={'s': 5, 'edgecolor': 'w', 'linewidth': .3}, stat_func=None)

    x_lim, y_lim = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
    g.ax_joint.plot(x_lim, y_lim, ls='--', lw=.3, c=bipal_dbgd[0])
    g.ax_joint.axhline(0.5, ls='-', lw=.1, c=bipal_dbgd[0])
    g.ax_joint.axvline(0.5, ls='-', lw=.1, c=bipal_dbgd[0])

    g.ax_joint.set_xlim(x_lim)
    g.ax_joint.set_ylim(y_lim)

    g.set_axis_labels('{} fold-changes AURCs'.format(x), '{} fold-changes AURCs'.format(y))

    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


if __name__ == '__main__':
    # - Import
    # Hart et al (non)essential genes
    essential = pd.read_csv(mp.HART_ESSENTIAL, sep='\t')['gene'].rename('essential')
    nessential = pd.read_csv(mp.HART_NON_ESSENTIAL, sep='\t')['gene'].rename('non-essential')

    # CRISPR library
    lib = pd.read_csv(mp.LIBRARY, index_col=0).groupby('gene')['chr'].first()

    # Samplesheet
    ss = pd.read_csv(mp.SAMPLESHEET, index_col=0)

    # Non-expressed genes
    nexp = pd.read_csv(mp.NON_EXP, index_col=0)

    # Copy-number
    cnv = pd.read_csv(mp.CN_GENE.format('snp'), index_col=0)

    # Ploidy
    ploidy = pd.read_csv(mp.CN_PLOIDY.format('snp'), index_col=0, names=['sample', 'ploidy'])['ploidy']

    # Chromosome copies
    chrm = pd.read_csv(mp.CN_CHR.format('snp'), index_col=0)

    # CRISPR
    c_gdsc_fc = pd.read_csv(mp.CRISPR_GENE_FC, index_col=0)
    c_gdsc_crispy = pd.read_csv(mp.CRISPR_GENE_CORRECTED_FC, index_col=0)

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(c_gdsc_crispy).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Plot: Cancer type coverage
    tissue_coverage(
        'Counts', 'Cancer type',
        ss.loc[samples, 'Cancer Type'].value_counts(ascending=True).reset_index().rename(columns={'index': 'Cancer type', 'Cancer Type': 'Counts'}),
        'reports/crispy/cancer_types_histogram.png'
    )

    # - Plot: QC Gene-sets AUCs
    auc_stats_fc_ess = auc_curves(c_gdsc_fc[samples], essential, 'reports/crispy/qc_aucs_fc_essential.png')
    auc_stats_fc_ness = auc_curves(c_gdsc_fc[samples], nessential, 'reports/crispy/qc_aucs_fc_nonessential.png')

    auc_stats_crispy_ess = auc_curves(c_gdsc_crispy[samples], essential, 'reports/crispy/qc_aucs_fc_corrected_essential.png')
    auc_stats_crispy_ness = auc_curves(c_gdsc_crispy[samples], nessential, 'reports/crispy/qc_aucs_fc_corrected_nonessential.png')

    aucs_scatter(
        'Original', 'Corrected',
        pd.concat([pd.Series(auc_stats_fc_ess['auc'], name='Original'), pd.Series(auc_stats_crispy_ess['auc'], name='Corrected')], axis=1),
        'reports/crispy/qc_aucs_essential.png', 'Essential genes'
    )

    aucs_scatter(
        'Original', 'Corrected',
        pd.concat([pd.Series(auc_stats_fc_ness['auc'], name='Original'), pd.Series(auc_stats_crispy_ness['auc'], name='Corrected')], axis=1),
        'reports/crispy/qc_aucs_nonessential.png', 'Non-essential genes'
    )

    # - Plot: Copy-number bias in ORIGINAL CRISPR fold-changes (AUCs)
    plot_df = pd.concat([
        pd.DataFrame({c: c_gdsc_fc[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples}).unstack().rename('fc'),
        cnv[samples].applymap(lambda v: bin_cnv(v, 10)).unstack().rename('cnv')
    ], axis=1).dropna()

    plot_df = plot_df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
    plot_df = plot_df[~plot_df['cnv'].isin(['0', '1'])]
    plot_df = plot_df.assign(chr=lib.loc[plot_df['gene']].values)

    hue_thresholds = natsorted(set(plot_df['cnv']))
    hue_thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(hue_thresholds)).as_hex()

    sns.boxplot('cnv', 'fc', data=plot_df, notch=True, sym='', order=hue_thresholds, palette=hue_thresholds_color, linewidth=.3)

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')
    plt.xlabel('Copy-number')
    plt.ylabel('CRISPR-Cas9 FC (log2)')
    plt.axhline(0., ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
    plt.gcf().set_size_inches(3, 3)

    plt.savefig('reports/crispy/copynumber_bias_boxplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Plot overall CNV bias
    copy_number_bias_aucs(plot_df, 'reports/crispy/copynumber_bias_aucs.png')

    plot_df_aucs = pd.DataFrame([{
        'sample': s, 'auc': a, 'cnv': t, 'ploidy': bin_cnv(ploidy[s], 5)
    } for s in set(plot_df['sample']) for t, a in copy_number_bias_aucs(plot_df[plot_df['sample'] == s])[1].items()])

    copy_number_bias_aucs_per_sample('cnv', 'auc', 'ploidy', plot_df_aucs, 'reports/crispy/copynumber_bias_aucs_per_sample.png')

    # Plot CNV bias per chromosome
    plot_df_aucs_chr = pd.DataFrame([{
        'sample': s, 'chr': c, 'auc': a, 'cnv': t, 'ploidy': bin_cnv(ploidy[s], 5), 'chr_cnv': bin_cnv(chrm.loc[c, s], 6)
    } for s in set(plot_df['sample']) for c in set(plot_df['chr']) for t, a in copy_number_bias_aucs(plot_df.query("(sample == '{}') & (chr == '{}')".format(s, c)))[1].items()])

    copy_number_bias_aucs_per_chrm('cnv', 'auc', 'chr_cnv', plot_df_aucs_chr.query("chr_cnv != '1'"), 'reports/crispy/copynumber_bias_aucs_per_chr.png')

    # - Plot: Copy-number bias in CORRECTED CRISPR fold-changes (AROCs)
    plot_df_crispy = pd.concat([
        pd.DataFrame({c: c_gdsc_crispy[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples if c in nexp}).unstack().rename('fc'),
        cnv[samples].applymap(lambda v: bin_cnv(v, 10)).unstack().rename('cnv')
    ], axis=1).dropna()

    plot_df_crispy = plot_df_crispy.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
    plot_df_crispy = plot_df_crispy[~plot_df_crispy['cnv'].isin(['0', '1'])]
    plot_df_crispy = plot_df_crispy.assign(chr=lib.loc[plot_df_crispy['gene']].values)

    # Plot overall CNV bias
    copy_number_bias_aucs(plot_df_crispy, 'reports/crispy/copynumber_bias_aucs_corrected.png')

    # Plot CNV bias per samples
    plot_df_crispy_aucs = pd.DataFrame([{
        'sample': s, 'auc': a, 'cnv': t, 'ploidy': bin_cnv(ploidy[s], 5)
    } for s in set(plot_df_crispy['sample']) for t, a in copy_number_bias_aucs(plot_df_crispy[plot_df_crispy['sample'] == s])[1].items()])

    copy_number_bias_aucs_per_sample('cnv', 'auc', 'ploidy', plot_df_crispy_aucs, 'reports/crispy/copynumber_bias_aucs_corrected_per_sample.png')

    # Plot CNV bias per chromosome
    plot_df_aucs_chr_crispy = pd.DataFrame([{
        'sample': s, 'chr': c, 'auc': a, 'cnv': t, 'ploidy': bin_cnv(ploidy[s], 5), 'chr_cnv': bin_cnv(chrm.loc[c, s], 6)
    } for s in set(plot_df_crispy['sample']) for c in set(plot_df_crispy['chr']) for t, a in copy_number_bias_aucs(plot_df_crispy.query("(sample == '{}') & (chr == '{}')".format(s, c)))[1].items()])

    copy_number_bias_aucs_per_chrm('cnv', 'auc', 'chr_cnv', plot_df_aucs_chr_crispy.query("chr_cnv != '1'"), 'reports/crispy/copynumber_bias_aucs_corrected_per_chr.png')

