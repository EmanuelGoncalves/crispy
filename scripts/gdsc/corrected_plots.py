#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import crispy as cy
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted


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


if __name__ == '__main__':
    # - Imports
    # Essential/non-essential aurcs
    qc_aucs = pd.read_csv(f'{gdsc.DIR}/qc_aucs.csv')

    # Copy-number aurcs
    cn_bias = pd.read_csv(f'{gdsc.DIR}/bias_aucs.csv')

    # - Essential AURCs comparison
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
    plt.savefig(f'reports/qc_gdsc_essential.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Copy-number bias comparison
    copynumber_bias_comparison(cn_bias)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/qc_gdsc_copynumber_bias.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
