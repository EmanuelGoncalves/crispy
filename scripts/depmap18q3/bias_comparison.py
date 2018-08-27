#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import depmap18q3
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import pearsonr


def aupr_comparison(aucs):
    plot_df = aucs.query("groupby == 'gene'").query("feature == 'ratio'")

    #
    order = natsorted(set(plot_df['feature_value']))
    pal = pd.Series(cy.QCplot.get_palette_continuous(len(order)), index=order)

    #
    aucs_mean = pd.pivot_table(
        plot_df, index='feature_value', columns='fold_change', values='auc', aggfunc=np.median
    ).loc[order]
    aucs_mean = aucs_mean.assign(copy_number=aucs_mean.index)

    aucs_perc_down = pd.pivot_table(
        plot_df, index='feature_value', columns='fold_change', values='auc', aggfunc=lambda v: np.percentile(v, 25)
    ).loc[order]
    aucs_perc_down = aucs_perc_down.assign(copy_number=aucs_perc_down.index)

    aucs_perc_up = pd.pivot_table(
        plot_df, index='feature_value', columns='fold_change', values='auc', aggfunc=lambda v: np.percentile(v, 75)
    ).loc[order]
    aucs_perc_up = aucs_perc_up.assign(copy_number=aucs_perc_up.index)

    #
    for i in reversed(order):
        x, y = aucs_mean.loc[i, 'ceres'], aucs_mean.loc[i, 'corrected']

        y_err_min, y_err_max = aucs_perc_down.loc[i, 'corrected'], aucs_perc_up.loc[i, 'corrected']
        yerr = np.array([y - y_err_min, y_err_max - y]).reshape(2, -1)

        x_err_min, x_err_max = aucs_perc_down.loc[i, 'ceres'], aucs_perc_up.loc[i, 'ceres']
        xerr = np.array([x - x_err_min, x_err_max - x]).reshape(2, -1)

        plt.errorbar(x, y, yerr=yerr, xerr=xerr, c=pal[i], label=i, fmt='o', lw=.3)

    plt.xlim((None, .8))
    plt.ylim((None, .8))

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.axhline(0.5, ls=':', lw=.5, color=cy.QCplot.PAL_DBGD[2], zorder=0)
    plt.axvline(0.5, ls=':', lw=.5, color=cy.QCplot.PAL_DBGD[2], zorder=0)

    plt.legend(loc='center left', title='Copy-number\nratio', bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 6})\
        .get_title().set_fontsize('7')

    plt.xlabel('CERES\ncopy-number AURC')
    plt.ylabel('Crispy\ncopy-number AURC')
    plt.title('Copy-number enrichment comparison\n(BROAD DepMap18Q3)')


def similarity_original(beds_gene):
    plot_df = pd.DataFrame({s: beds_gene[s].corr().iloc[0, :] for s in beds_gene}).drop(['fold_change']).T

    plt.scatter(plot_df['ceres'], plot_df['corrected'], c=cy.QCplot.PAL_DBGD[0], alpha=.7, lw=.1, edgecolors='white')

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.xlabel('CERES\ncorrelation original')
    plt.ylabel('Crispy\ncorrelation original')
    plt.title('Similarity with original fold-changes\n(BROAD DepMap18Q3)')


def essential_aurc(beds_gene):
    essential = cy.Utils.get_essential_genes(return_series=False)

    ess_aurcs = pd.DataFrame({
        s: {f: cy.QCplot.recall_curve(beds_gene[s][f], essential)[2] for f in list(beds_gene[s])} for s in beds_gene
    }).T

    return ess_aurcs


def essential_recall_comparison(ess_aurcs):
    plt.scatter(ess_aurcs['ceres'], ess_aurcs['corrected'], color=cy.QCplot.PAL_DBGD[0], alpha=.7, lw=.1, edgecolors='white')

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.xlabel('CERES\nessential genes AURC')
    plt.ylabel('Crispy\nessential genes AURC')
    plt.title('Essential genes recall\n(BROAD DepMap18Q3)')


def essential_replicates(ess_aurcs):
    plot_df = pd.concat([
        ess_aurcs.eval('ceres - corrected').rename('ceres'),
        rep_corr.groupby('name_1')['corr'].mean()
    ], axis=1, sort=False).dropna()

    scatter_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6)
    line_kws = dict(lw=1., color=cy.QCplot.PAL_DBGD[1], alpha=1.)
    joint_kws = dict(lowess=False, scatter_kws=scatter_kws, line_kws=line_kws)
    marginal_kws = dict(kde=False, hist_kws={'linewidth': 0})
    annot_kws = dict(stat='R')

    g = sns.jointplot(
        'ceres', 'corr', data=plot_df, kind='reg', space=0, color=cy.QCplot.PAL_DBGD[0],
        marginal_kws=marginal_kws, annot_kws=annot_kws, joint_kws=joint_kws
    )

    g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}', frameon=False)

    g.set_axis_labels('CERES\nessential genes recall gain', 'Replicates correlation\n(mean Pearson)')


if __name__ == '__main__':
    # - Imports
    # Manifest
    samplesheet = depmap18q3.get_samplesheet()

    manifest = depmap18q3.get_replicates_map().dropna()
    manifest = manifest.replace({'Broad_ID': samplesheet['CCLE_name']})
    manifest = manifest.set_index('replicate_ID')

    # Crispy bed files
    beds = depmap18q3.import_crispy_beds()

    # Replicates correlation
    rep_corr = pd.read_csv(f'{depmap18q3.DIR}/replicates_foldchange_correlation.csv')
    rep_corr = rep_corr.assign(name_1=manifest.loc[rep_corr['sample_1'], 'Broad_ID'].values)
    rep_corr = rep_corr.assign(name_2=manifest.loc[rep_corr['sample_2'], 'Broad_ID'].values)
    rep_corr = rep_corr[rep_corr['name_1'] == rep_corr['name_2']]

    # Add ceres gene scores to bed
    ceres = depmap18q3.get_ceres()
    beds = {s: beds[s].assign(ceres=ceres.loc[beds[s]['gene'], s].values) for s in beds}

    # Copy-number bias AURCs
    aucs = pd.read_csv(f'{depmap18q3.DIR}/bias_aucs.csv')

    # - Aggregate by genes
    agg_fun = dict(fold_change=np.mean, corrected=np.mean, ceres=np.mean)
    beds_gene = {s: beds[s].groupby(['gene']).agg(agg_fun) for s in beds}

    # - Essential AURCs
    ess_aurcs = essential_aurc(beds_gene)

    # - Essential genes recall capacity comparison
    essential_recall_comparison(ess_aurcs)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/bias_essentiality_comparison_depmap18q3.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Essential genes recall gain comparison with replicates correlation
    essential_replicates(ess_aurcs)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/bias_essentiality_replicates_ceres_depmap18q3.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Similarity of corrected fold-changes compared to original fold-changes
    similarity_original(beds_gene)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/bias_similarity_comparison_depmap18q3.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Comparison of correction AURCs
    aupr_comparison(aucs)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/bias_aucs_comparison_depmap18q3.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Gene copy-number ratio bias boxplots CERES
    plot_df = aucs.query("groupby == 'gene'").query("feature == 'ratio'").query("fold_change == 'ceres'")

    ax = cy.QCplot.bias_boxplot(plot_df, x='feature_value', y='auc', notch=False, add_n=True, n_text_y=.05)

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number ratio')
    plt.ylabel(f'AURC (per cell line)')
    plt.title('Copy-number effect on CRISPR-Cas9\nCERES corrected (BROAD DepMap18Q3)')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig(f'reports/bias_aucs_comparison_depmap18q3_ceres_boxplots.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Gene copy-number ratio bias boxplots Crispy
    plot_df = aucs.query("groupby == 'gene'").query("feature == 'ratio'").query("fold_change == 'corrected'")

    ax = cy.QCplot.bias_boxplot(plot_df, x='feature_value', y='auc', notch=False, add_n=True, n_text_y=.05)

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number ratio')
    plt.ylabel(f'AURC (per cell line)')
    plt.title('Copy-number effect on CRISPR-Cas9\nCrispy corrected (BROAD DepMap18Q3)')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig(f'reports/bias_aucs_comparison_depmap18q3_crispy_boxplots.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
