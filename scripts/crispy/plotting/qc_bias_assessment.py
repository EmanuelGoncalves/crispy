#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from natsort import natsorted
from crispy.utils import bin_cnv
from crispy.benchmark_plot import plot_cumsum_auc
from sklearn.metrics import roc_curve, auc, roc_auc_score


def auc_curves(df, geneset, outfile):
    # Palette
    pal = sns.light_palette(bipal_dbgd[0], n_colors=df.shape[1]).as_hex()

    # AUCs fold-changes
    ax, stats_ess = plot_cumsum_auc(df, geneset, plot_mean=True, legend=False, palette=pal)
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

    ax.set_xlabel('{} fold-changes AUCs'.format(x))
    ax.set_ylabel('{} fold-changes AUCs'.format(y))
    ax.set_title(title)

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def copy_number_bias_arocs(y_true, y_pred, data, outfile):
    thresholds = natsorted(set(data[y_true]))
    thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(thresholds)).as_hex()

    ax = plt.gca()

    for c, thres in zip(thresholds_color, thresholds):
        fpr, tpr, _ = roc_curve((data[y_true] == thres).astype(int), -data[y_pred])
        ax.plot(fpr, tpr, label='%s: AUC=%.2f' % (thres, auc(fpr, tpr)), lw=1., c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
    legend = ax.legend(loc=4, title='Copy-number', prop={'size': 6})
    legend.get_title().set_fontsize('7')

    # plt.show()
    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


def copy_number_bias_arocs_per_sample(x, y, hue, data, outfile):
    thresholds = natsorted(set(data[x]))
    thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(thresholds)).as_hex()

    # Plot per sample
    sns.boxplot(x, y, data=data, sym='', order=thresholds, palette=thresholds_color, linewidth=.3)
    sns.stripplot(x, y, data=data, order=thresholds, palette=thresholds_color, edgecolor='white', linewidth=.1, size=3, jitter=.4)

    plt.title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AUC (cell lines)')
    plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Plot per sample hue by ploidy
    hue_thresholds = natsorted(set(data[hue]))
    hue_thresholds_color = sns.light_palette(bipal_dbgd[0], n_colors=len(hue_thresholds)).as_hex()

    sns.boxplot(x, y, hue, data=data, sym='', order=thresholds, hue_order=hue_thresholds, palette=hue_thresholds_color, linewidth=.3)
    sns.stripplot(x, y, hue, data=data, order=thresholds, hue_order=hue_thresholds, palette=hue_thresholds_color, edgecolor='white', linewidth=.1, size=1, jitter=.2, split=True)

    handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(hue_thresholds_color, hue_thresholds))]
    plt.legend(handles=handles, title='Ploidy', prop={'size': 6}).get_title().set_fontsize('6')
    plt.title('Cell ploidy bias on CRISPR/Cas9\nnon-expressed genes')
    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AUC (cell lines)')
    plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
    plt.gcf().set_size_inches(3, 3)

    outfile_dir, outfile_exp = os.path.splitext(outfile)
    plt.savefig('{}_ploidy.png'.format(outfile_dir, outfile_exp), bbox_inches='tight', dpi=600)
    plt.close('all')


def main():
    # - Import
    # (non)essential genes
    essential = pd.read_csv('data/gene_sets/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')
    nessential = pd.read_csv('data/gene_sets/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

    # Non-expressed genes
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

    # Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0)

    # Ploidy
    ploidy = pd.read_csv('data/crispy_copy_number_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

    # CRISPR
    c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)
    c_gdsc_crispy = pd.read_csv('data/crispr_gdsc_logfc_corrected.csv', index_col=0)

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(c_gdsc_crispy).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

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

    # - Plot: Copy-number bias AROCs
    plot_df = pd.concat([
        pd.DataFrame({c: c_gdsc_fc.reindex(nexp[c])[c] for c in samples if c in nexp}).unstack().rename('fc'),
        cnv[samples].applymap(lambda v: bin_cnv(v, 10)).unstack().rename('cnv')
    ], axis=1).dropna()
    plot_df = plot_df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
    plot_df = plot_df[~plot_df['cnv'].isin(['0', '1'])]

    # Plot overall CNV bias
    copy_number_bias_arocs('cnv', 'fc', plot_df, 'reports/crispy/copynumber_bias.png')

    # Plot CNV bias per samples
    plot_df_aucs = pd.DataFrame({
        c: {
            t: roc_auc_score(
                (plot_df.query("sample == '{}'".format(c))['cnv'] == t).astype(int), -plot_df.query("sample == '{}'".format(c))['fc']
            ) for t in set(plot_df['cnv']) if (sum(plot_df.query("sample == '{}'".format(c))['cnv'] == t) >= 5) and (len(set(plot_df.query("sample == '{}'".format(c))['fc'])) > 1)
        } for c in set(plot_df['sample'])
    })
    plot_df_aucs = plot_df_aucs.unstack().reset_index().dropna().rename(columns={'level_0': 'sample', 'level_1': 'cnv', 0: 'auc'})
    plot_df_aucs = plot_df_aucs.assign(ploidy=ploidy.loc[plot_df_aucs['sample']].apply(lambda v: bin_cnv(v, 5)).values)

    copy_number_bias_arocs_per_sample('cnv', 'auc', 'ploidy', plot_df_aucs, 'reports/crispy/copynumber_bias_per_sample.png')


if __name__ == '__main__':
    main()
