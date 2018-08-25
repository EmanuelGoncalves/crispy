#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import depmap18q3 as depmap18q3
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Import crispy bed files
    beds = depmap18q3.import_crispy_beds()

    # - Add ceres gene scores to bed
    ceres = depmap18q3.get_ceres()
    beds = {s: beds[s].assign(ceres=ceres.loc[beds[s]['gene'], s].values) for s in beds}

    # - Aggregate by segments
    agg_fun = dict(
        fold_change=np.mean, gp_mean=np.mean, corrected=np.mean, ceres=np.mean,
        copy_number=np.mean, chr_copy=np.mean, ploidy=np.mean, ratio=np.mean, len=np.mean
    )

    beds_seg = {s: beds[s].groupby(['chr', 'start', 'end']).agg(agg_fun) for s in beds}

    # - Aggregate by genes
    agg_fun = dict(
        fold_change=np.mean, gp_mean=np.mean, corrected=np.mean, ceres=np.mean, copy_number=np.mean, ratio=np.mean
    )

    beds_gene = {s: beds[s].groupby(['gene']).agg(agg_fun) for s in beds}

    # - Copy-number bias
    aucs_df = []
    for y_label, y_var in [('original', 'fold_change'), ('corrected', 'corrected'), ('ceres', 'ceres')]:

        for x_var, x_thres in [('copy_number', 10), ('ratio', 10)]:

            for groupby, df in [('segment', beds_seg), ('gene', beds_gene)]:
                # Calculate bias
                cn_aucs_df = pd.concat([
                    cy.QCplot.recall_curve_discretise(
                        df[s][y_var], df[s][x_var], x_thres
                    ).assign(sample=s) for s in df
                ])

                # Plot
                ax = cy.QCplot.bias_boxplot(cn_aucs_df, x=x_var, notch=False, add_n=True, n_text_y=.05)

                ax.set_ylim(0, 1)
                ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

                x_label = x_var.replace('_', ' ').capitalize()
                plt.xlabel(f'{x_label}')

                plt.ylabel(f'AURC (per {groupby})')

                plt.title('Copy-number effect on CRISPR-Cas9')

                plt.gcf().set_size_inches(3, 3)
                plt.savefig(f'reports/depmap_18q3/bias_copynumber_{groupby}_{y_label}_{x_label}.png', bbox_inches='tight', dpi=600)
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
    aucs_df.to_csv(f'{depmap18q3.DIR}/bias_aucs.csv', index=False)

    # -
    order = ['Original', 'Crispy', 'CERES']

    for feature in ['copy_number', 'ratio']:
        for groupby in ['gene', 'segment']:
            # Build data-frame
            plot_df = aucs_df.query('feature == @feature & groupby == @groupby')
            plot_df = pd.pivot_table(plot_df, index=['sample', 'feature_value'], columns='fold_change', values='auc')
            plot_df = plot_df.rename(columns={'original': 'Original', 'corrected': 'Crispy', 'ceres': 'CERES'})

            # Plot
            cy.QCplot.aucs_scatter_pairgrid(plot_df[order])

            feature_label = feature.replace('_', '').capitalize()
            plt.suptitle(f'{feature_label} AURC (by {groupby})', y=1.03)

            plt.gcf().set_size_inches(4, 4)
            plt.savefig(f'reports/depmap_18q3/bias_pairgrid_{groupby}_{feature}.png', bbox_inches='tight', dpi=600)
            plt.close('all')
