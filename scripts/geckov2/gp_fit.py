#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import geckov2
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF, Matern, RationalQuadratic


if __name__ == '__main__':
    # -
    # CERES
    ceres = geckov2.get_ceres()

    # Gene-sets
    essential, nessential = cy.Utils.get_essential_genes(), cy.Utils.get_non_essential_genes()

    # -
    beds = {}
    for s in ceres:
        bed_df = pd.read_csv(f'{geckov2.DIR}/bed/{s}.crispy.bed', sep='\t')
        bed_df = bed_df.assign(ceres=ceres.reindex(bed_df['gene'])[s].values).dropna()
        beds[s] = bed_df

    # -
    scores = []
    for f_name, features in [('copy_number', ['copy_number']), ('ratio', ['ratio']), ('all', ['ratio', 'copy_number', 'chr_copy', 'len_log2'])]:
        for s in ceres:
            cv = ShuffleSplit(n_splits=10, test_size=.3)

            for train_idx, test_idx in cv.split(cy.CrispyGaussian(beds[s], n_sgrna=0).bed_seg):
                kernel = ConstantKernel() * RBF() + WhiteKernel()

                gpr = cy.CrispyGaussian(beds[s], kernel=kernel, n_sgrna=0).fit(features, 'fold_change', train_idx=train_idx)

                r2 = gpr.score(x=gpr.bed_seg.iloc[test_idx][features], y=gpr.bed_seg.iloc[test_idx]['fold_change'])

                scores.append(dict(sample=s, feature=f_name, score=r2))

    # -
    scores = pd.DataFrame(scores)

    #
    plot_df = pd.pivot_table(scores, index='sample', columns=['feature'], values='score', aggfunc=np.mean)

    x, y = 'copy_number', 'all'
    ax = cy.QCplot.aucs_scatter(x, y, data=plot_df, s=10, marker='o', lw=.3)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/geckov2/gp_fit_feature_type.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    order = ['Original', 'Crispy', 'CERES']

    # -
    gset_aucs = []
    for gset in [nessential, essential]:
        for y_var in order:
            for s in gene_foldchange:
                _, _, auc = cy.QCplot.recall_curve(gene_foldchange[s][y_var], gset)
                gset_aucs.append(dict(sample=s, aurc=auc, type=y_var, geneset=gset.name))

    gset_aucs = pd.DataFrame(gset_aucs)

    #
    for gset in [essential, nessential]:
        # Build data-frame
        plot_df = gset_aucs.query(f"geneset == '{gset.name}'")
        plot_df = pd.pivot_table(plot_df, index='sample', columns='type', values='aroc')

        # Plot
        cy.QCplot.aucs_scatter_pairgrid(plot_df[order], set_lim=False, axhlines=False, axvlines=False)

        plt.suptitle(f'{gset.name}', y=1.03)

        plt.gcf().set_size_inches(4, 4)
        plt.savefig(f'reports/geckov2/gp_fit_{gset.name}.png', bbox_inches='tight', dpi=600)
        plt.close('all')

    # -
    cn_aucs = []
    for y_var in order:
        df = pd.concat([
            cy.QCplot.recall_curve_discretise(
                gene_foldchange[s][y_var], gene_foldchange[s]['copy_number'], 10
            ).assign(sample=s).assign(type=y_var) for s in gene_foldchange
        ])

        cn_aucs.append(df)

    cn_aucs = pd.concat(cn_aucs)

    # Build data-frame
    plot_df = pd.pivot_table(cn_aucs[~cn_aucs.isin(['0', '1', '2', '3', '4'])], index=['sample', 'copy_number'], columns='type', values='auc')

    # Plot
    cy.QCplot.aucs_scatter_pairgrid(plot_df[order])

    plt.suptitle(f'Copy-number', y=1.03)

    plt.gcf().set_size_inches(4, 4)
    plt.savefig(f'reports/geckov2/gp_fit_copynumber.png', bbox_inches='tight', dpi=600)
    plt.close('all')
