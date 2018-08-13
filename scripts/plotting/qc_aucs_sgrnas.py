#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
from sklearn.metrics import roc_auc_score

import crispy as cy
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from plotting.qc_aucs import aucs_scatter
from scripts.plotting import get_palette_continuous


def aucs_curves(df, geneset):
    pal = get_palette_continuous(df.shape[1])

    ax, auc_stats = cy.plot_cumsum_auc(df, geneset, plot_mean=False, legend=False, palette=pal)

    mean_auc = pd.Series(auc_stats['auc']).mean()

    plt.title('CRISPR-Cas9 {} fold-changes\nmean AURC = {:.2f}'.format(df.columns.name, mean_auc))

    return auc_stats


if __name__ == '__main__':
    # - Imports
    # CRISPR library
    lib = mp.get_crispr_lib()

    # Ploidy
    ploidy = mp.get_ploidy()

    # CRISPR
    crispr = mp.get_crispr_sgrna(dropna=False, scale=False)
    crispr = crispr.loc[:, crispr.count() == 100212].dropna()
    crispr.columns.name = 'original'

    # Gene-sets
    essential = pd.Series(lib[lib['GENES'].isin(mp.get_essential_genes())].index).rename('essential')
    nessential = pd.Series(lib[lib['GENES'].isin(mp.get_non_essential_genes())].index).rename('non-essential')
    control = pd.Series([i for i in crispr.index if i.startswith('CTRL0')]).rename('non-targeting')

    #
    aucs = []
    for gset in [essential, nessential, control]:
        df_aucs = aucs_curves(crispr, gset)

        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f'reports/qc_aucs_sgrnas_{gset.name}.png', bbox_inches='tight', dpi=600)
        plt.close('all')

        aucs.append(pd.DataFrame(dict(aurc=df_aucs['auc'], geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})

    # -
    genes_guides = lib.reindex(crispr.index)['GENES'].dropna()

    # CRISPR gene
    crispr_genes = crispr.groupby(genes_guides).mean()
    crispr_genes.columns.name = 'mean'

    # Average without non working guides
    crispr_opt, n_bad_guides = {}, {}
    for s in crispr:
        nt_fc = crispr.loc[control, s].mean()

        good_guides = np.logical_and(crispr.loc[genes_guides.index, s] > nt_fc, crispr_genes.loc[genes_guides.values, s] < 0)
        n_bad_guides[s] = sum(good_guides)

        good_guides = good_guides[~good_guides]
        good_guides = genes_guides.loc[good_guides.index]

        crispr_opt[s] = crispr.loc[good_guides.index, s].groupby(good_guides).mean()

    crispr_opt = pd.DataFrame(crispr_opt)
    crispr_opt.columns.name = 'optimised'

    n_bad_guides = pd.Series(n_bad_guides).sort_values()

    # Gene-sets
    genes_essential = pd.Series(list(mp.get_essential_genes())).rename('essential')
    genes_nessential = pd.Series(list(mp.get_non_essential_genes())).rename('non-essential')

    rocs_genes = []
    for df in [crispr_genes, crispr_opt]:
        df = df.reindex(genes_essential.append(genes_nessential)).dropna()

        y_true = df.index.isin(genes_essential).astype(int)

        for s in df:
            s_aucs = roc_auc_score(y_true, -df[s])
            rocs_genes.append(dict(
                aurc=s_aucs, type=df.columns.name, sample=s
            ))

    aucs_genes = pd.DataFrame(rocs_genes)

    # Scatter comparison
    plot_df = pd.pivot_table(aucs_genes, index='sample', columns='type', values='aurc')

    ax = aucs_scatter('mean', 'optimised', plot_df)

    ax.set_title('AURCs genes')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/qc_arocs_optimised.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # aucs_genes = []
    # for df in [crispr_genes, crispr_opt]:
    #     for gset in [genes_essential, genes_nessential]:
    #         df_aucs = aucs_curves(df, gset)
    #
    #         plt.gcf().set_size_inches(3, 3)
    #         plt.savefig(f'reports/qc_aucs_optimised_{df.columns.name}_{gset.name}.png', bbox_inches='tight', dpi=600)
    #         plt.close('all')
    #
    #         aucs_genes.append(pd.DataFrame(dict(aurc=df_aucs['auc'], type=df.columns.name, geneset=gset.name)))
    #
    # aucs_genes = pd.concat(aucs_genes).reset_index().rename(columns={'index': 'sample'})
    #
    # # - Scatter comparison
    # for gset in [genes_essential, genes_nessential]:
    #     plot_df = pd.pivot_table(
    #         aucs_genes.query("geneset == '{}'".format(gset.name)),
    #         index='sample', columns='type', values='aurc'
    #     )
    #
    #     ax = aucs_scatter('mean', 'optimised', plot_df)
    #
    #     ax.set_title('AURCs {} genes'.format(gset.name))
    #
    #     plt.gcf().set_size_inches(3, 3)
    #     plt.savefig('reports/qc_aucs_optimised_{}.png'.format(gset.name), bbox_inches='tight', dpi=600)
    #     plt.close('all')
    #
