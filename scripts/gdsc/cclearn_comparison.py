#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import pandas as pd
import crispy as cy
from natsort import natsorted
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Crispy bed files
    beds = gdsc.import_crispy_beds()

    # - Add ceres gene scores to bed
    ccleanr = gdsc.import_ccleanr()

    # - Aggregate by gene
    agg_fun = dict(
        fold_change=np.mean, copy_number=np.mean, ploidy=np.mean, chr_copy=np.mean, ratio=np.mean, chr='first',
        corrected=np.mean
    )

    bed_gene = []

    for s in beds:
        df = beds[s].groupby('gene').agg(agg_fun)
        df = df.assign(scaled=cy.Crispy.scale_crispr(df['fold_change']))
        df = df.assign(ccleanr=ccleanr[s].reindex(df.index))
        df = df.assign(sample=s)

        bed_gene.append(df)

    bed_gene = pd.concat(bed_gene).reset_index()

    # -
    ccleanr_aucs = pd.concat([cy.QCplot.recall_curve_discretise(
        bed_gene.query('sample == @s')[fc],
        bed_gene.query('sample == @s')['ratio'],
        4
    ).assign(sample=s).assign(feature=fc) for s in beds for fc in ['ccleanr', 'corrected']])

    # -
    order = natsorted(set(ccleanr_aucs['ratio']))
    pal = pd.Series(cy.QCplot.get_palette_continuous(len(order)), index=order)

    #
    aucs_mean = pd.pivot_table(
        ccleanr_aucs, index='ratio', columns='feature', values='auc', aggfunc=np.median
    ).loc[order]
    aucs_mean = aucs_mean.assign(copy_number=aucs_mean.index)

    aucs_perc_down = pd.pivot_table(
        ccleanr_aucs, index='ratio', columns='feature', values='auc', aggfunc=lambda v: np.percentile(v, 25)
    ).loc[order]
    aucs_perc_down = aucs_perc_down.assign(copy_number=aucs_perc_down.index)

    aucs_perc_up = pd.pivot_table(
        ccleanr_aucs, index='ratio', columns='feature', values='auc', aggfunc=lambda v: np.percentile(v, 75)
    ).loc[order]
    aucs_perc_up = aucs_perc_up.assign(copy_number=aucs_perc_up.index)

    #
    for i in reversed(order):
        x, y = aucs_mean.loc[i, 'ccleanr'], aucs_mean.loc[i, 'corrected']

        y_err_min, y_err_max = aucs_perc_down.loc[i, 'corrected'], aucs_perc_up.loc[i, 'corrected']
        yerr = np.array([y - y_err_min, y_err_max - y]).reshape(2, -1)

        x_err_min, x_err_max = aucs_perc_down.loc[i, 'ccleanr'], aucs_perc_up.loc[i, 'ccleanr']
        xerr = np.array([x - x_err_min, x_err_max - x]).reshape(2, -1)

        plt.errorbar(x, y, yerr=yerr, xerr=xerr, c=pal[i], label=i, fmt='o', lw=.3)

    (x0, x1), (y0, y1) = plt.xlim(), plt.ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt.plot(lims, lims, lw=.5, zorder=0, ls=':', color=cy.QCplot.PAL_DBGD[2])

    plt.axhline(0.5, ls=':', lw=.5, color=cy.QCplot.PAL_DBGD[2], zorder=0)
    plt.axvline(0.5, ls=':', lw=.5, color=cy.QCplot.PAL_DBGD[2], zorder=0)

    plt.legend(loc='center left', title='Copy-number\nratio', bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 6})\
        .get_title().set_fontsize('7')

    plt.xlabel('CRISPRcleanR\ncopy-number ratio AURC')
    plt.ylabel('Crispy\ncopy-number AURC')
    plt.title('Copy-number enrichment comparison')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/bias_aucs_comparison_ccleanr.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
