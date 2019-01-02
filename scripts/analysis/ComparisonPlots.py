#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import pearsonr
from analysis.DataImporter import DataImporter


def estimate_copynumber_bias(beds):
    # - Aggregate by segments
    agg_fun = dict(
        fold_change=np.mean, gp_mean=np.mean, corrected=np.mean, ceres=np.mean,
        copy_number=np.mean, chr_copy=np.mean, ploidy=np.mean, ratio=np.mean, len=np.mean
    )

    beds_seg = {s: beds[s].groupby(['chr', 'start', 'end']).agg(agg_fun) for s in beds}

    # - Aggregate by genes
    agg_fun = dict(
        fold_change=np.mean, gp_mean=np.mean, corrected=np.mean, copy_number=np.mean, ratio=np.mean, ceres=np.mean
    )

    beds_gene = {s: beds[s].groupby(['gene']).agg(agg_fun) for s in beds}

    # - Copy-number bias
    aucs_df = []
    for y_label, y_var in [('ceres', 'ceres'), ('original', 'fold_change'), ('corrected', 'corrected')]:

        for x_var, x_thres in [('copy_number', 10), ('ratio', 4)]:

            for groupby, df in [('segment', beds_seg), ('gene', beds_gene)]:

                cn_aucs_df = pd.concat([
                    cy.QCplot.recall_curve_discretise(
                        df[s][y_var], df[s][x_var], x_thres
                    ).assign(sample=s) for s in df
                ])

                ax = cy.QCplot.bias_boxplot(cn_aucs_df, x=x_var, notch=False, add_n=True, n_text_y=.05)

                ax.set_ylim(0, 1)
                ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

                x_label = x_var.replace('_', ' ').capitalize()
                plt.xlabel(f'{x_label}')

                plt.ylabel(f'AURC (per {groupby})')

                plt.title('Copy-number effect on CRISPR-Cas9')

                plt.gcf().set_size_inches(3, 3)
                plt.savefig(f'reports/gdsc/bias_copynumber_{groupby}_{y_label}_{x_label}.png', bbox_inches='tight', dpi=600)
                plt.close('all')

                # Export
                cn_aucs_df = pd.melt(cn_aucs_df, id_vars=['sample', x_var], value_vars=['auc'])

                cn_aucs_df = cn_aucs_df \
                    .assign(fold_change=y_label) \
                    .assign(feature=x_var) \
                    .assign(groupby=groupby) \
                    .drop(['variable'], axis=1) \
                    .rename(columns={x_var: 'feature_value', 'value': 'auc'})

                aucs_df.append(cn_aucs_df)

    aucs_df = pd.concat(aucs_df)
    return aucs_df


def copynumber_bias_comparison(cn_bias):
    plot_df = cn_bias.query("groupby == 'gene'").query("feature == 'ratio'")

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
        x, y = aucs_mean.loc[i, 'original'], aucs_mean.loc[i, 'corrected']

        y_err_min, y_err_max = aucs_perc_down.loc[i, 'corrected'], aucs_perc_up.loc[i, 'corrected']
        yerr = np.array([y - y_err_min, y_err_max - y]).reshape(2, -1)

        x_err_min, x_err_max = aucs_perc_down.loc[i, 'original'], aucs_perc_up.loc[i, 'original']
        xerr = np.array([x - x_err_min, x_err_max - x]).reshape(2, -1)

        plt.errorbar(x, y, yerr=yerr, xerr=xerr, c=pal[i], label=i, fmt='o', lw=.3)

    plt.xlim((None, 1))
    plt.ylim((None, 1))

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.axhline(0.5, ls=':', lw=.5, color=cy.QCplot.PAL_DBGD[2], zorder=0)
    plt.axvline(0.5, ls=':', lw=.5, color=cy.QCplot.PAL_DBGD[2], zorder=0)

    plt.legend(title='Copy-number ratio', loc=2, frameon=False, prop={'size': 6})\
        .get_title().set_fontsize('6')

    plt.xlabel('Original fold-changes\ncopy-number AURC')
    plt.ylabel('Crispy corrected fold-changes\ncopy-number AURC')
    plt.title('Copy-number enrichment\ncorrection')


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
    plt.title('Essential genes recall')


def essential_replicates(ess_aurcs, rep_corr):
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


def similarity_original(beds_gene, xlim=None, ylim=None):
    plot_df = pd.DataFrame({s: beds_gene[s].corr().iloc[0, :] for s in beds_gene}).drop(['fold_change']).T

    plt.scatter(plot_df['ceres'], plot_df['corrected'], c=cy.QCplot.PAL_DBGD[0], alpha=.7, lw=.1, edgecolors='white')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.xlabel('CERES scores\ncorrelation with original')
    plt.ylabel('Crispy scores\ncorrelation with original')
    plt.title('Similarity with original fold-changes')


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

    plt.legend(title='Copy-number ratio', loc=2, frameon=False, prop={'size': 6})\
        .get_title().set_fontsize('6')

    plt.xlabel('CERES\ncopy-number AURC')
    plt.ylabel('Crispy\ncopy-number AURC')
    plt.title('Copy-number enrichment')


if __name__ == '__main__':
    # - Import data
    data = DataImporter()
    beds = data.get_crispy_beds()

    # - Add CERES gene scores to bed
    ceres = data.get_ceres_scores()
    beds = {s: beds[s].assign(ceres=ceres.loc[beds[s]['gene'], s].values) for s in beds}

    # - Replicates correlation
    replicates_corr = pd.read_csv('data/analysis/replicates_correlation_sgrna_fc.csv').query('replicate == 1')

    # - Bias
    aucs_df = estimate_copynumber_bias(beds)
    aucs_df.to_csv('data/analysis/copynumber_bias_aurc.csv', index=False)

    qc_aucs = pd.read_csv('data/analysis/gene_ess_ness_aucs.csv')

    # - Aggregate by genes
    agg_fun = dict(fold_change=np.mean, corrected=np.mean, ceres=np.mean)
    beds_gene = {s: beds[s].groupby(['gene']).agg(agg_fun) for s in beds}

    # - Essential AURCs
    ess_aurcs = essential_aurc(beds_gene)

    # - Plotting
    # Essential genes recall capacity comparison
    essential_recall_comparison(ess_aurcs)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/analysis/comparison_ceres_essentiality.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Essential genes recall gain comparison with replicates correlation
    essential_replicates(ess_aurcs, replicates_corr)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/analysis/comparison_ceres_replicates.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Similarity of corrected fold-changes compared to original fold-changes
    similarity_original(beds_gene, xlim=(0.85, 1.02), ylim=(0.85, 1.02))
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/analysis/comparison_ceres_similarity.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Comparison of correction AURCs
    aupr_comparison(aucs_df)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/analysis/comparison_ceres_aucs.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Gene copy-number ratio bias boxplots CERES
    plot_df = aucs_df.query("groupby == 'gene'").query("feature == 'ratio'").query("fold_change == 'ceres'")

    ax = cy.QCplot.bias_boxplot(plot_df, x='feature_value', y='auc', notch=True, add_n=True, n_text_y=.05)

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number ratio')
    plt.ylabel(f'AURC (per cell line)')
    plt.title('Copy-number effect on CRISPR-Cas9\nCERES corrected')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig(f'reports/analysis/comparison_aucs_ceres_boxplots.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Gene copy-number ratio bias boxplots Crispy
    plot_df = aucs_df.query("groupby == 'gene'").query("feature == 'ratio'").query("fold_change == 'corrected'")

    ax = cy.QCplot.bias_boxplot(plot_df, x='feature_value', y='auc', notch=True, add_n=True, n_text_y=.05)

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number ratio')
    plt.ylabel(f'AURC (per cell line)')
    plt.title('Copy-number effect on CRISPR-Cas9\nCrispy corrected')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig(f'reports/analysis/comparison_crispy_boxplots.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Essential AURCs comparison
    plot_df = qc_aucs.query("geneset == 'essential'")
    plot_df = pd.pivot_table(plot_df, index='sample', columns='type', values='aurc')

    plt.scatter(plot_df['original'], plot_df['corrected'], c=cy.QCplot.PAL_DBGD[0], alpha=.7, lw=.1, edgecolors='white')

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.xlabel('Original fold-changes')
    plt.ylabel('Crispy corrected fold-changes')
    plt.title('AURCs of essential genes')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/analysis/comparison_essential_aucs.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Copy-number bias comparison
    copynumber_bias_comparison(aucs_df)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/analysis/comparison_copynumber_bias.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    plot_df = pd.pivot_table(
        aucs_df.query("feature == 'ratio'"),
        index=['sample', 'feature_value'], columns=['fold_change'], values='auc'
    )

    plot_df = plot_df.eval('original - corrected').rename('diff').sort_values().to_frame().reset_index()
    plot_df['cancer_type'] = data.samplesheet.loc[plot_df['sample'], 'Cancer Type'].values

    #
    samples_replicates = pd.read_csv(f'data/analysis/replicates_correlation_gene_fc.csv').query('replicate == 1')

    supmaterial = pd.concat([
        data.samplesheet[['COSMIC ID', 'Tissue', 'Cancer Type', 'TCGA', 'CCLE ID']],
        pd.Series({c: beds[c]['ploidy'].mean().round(5) for c in data.samples}).rename('ploidy'),
        samples_replicates.groupby('name_1')['corr'].mean().rename('replicate_corr')
    ], axis=1, sort=False).sort_values('Tissue')
