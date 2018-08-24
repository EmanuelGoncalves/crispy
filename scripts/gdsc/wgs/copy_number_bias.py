#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import crispy as cy
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Import crispy bed files
    beds = gdsc.import_crispy_beds(wgs=True)

    # - Aggregate by segments
    agg_fun = dict(
        fold_change=np.mean, gp_mean=np.mean, corrected=np.mean,
        copy_number=np.mean, chr_copy=np.mean, ploidy=np.mean, ratio=np.mean, len=np.mean
    )

    beds_seg = {s: beds[s].groupby(['chr', 'start', 'end']).agg(agg_fun) for s in beds}

    # - Aggregate by genes
    agg_fun = dict(
        fold_change=np.mean, gp_mean=np.mean, corrected=np.mean, copy_number=np.mean, ratio=np.mean
    )

    beds_gene = {s: beds[s].groupby(['gene']).agg(agg_fun) for s in beds}

    # - Copy-number bias
    for y_label, y_var in [('original', 'fold_change'), ('corrected', 'corrected')]:

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
                plt.savefig(f'reports/gdsc/wgs/bias_copynumber_{groupby}_{y_label}_{x_label}.png', bbox_inches='tight', dpi=600)
                plt.close('all')
