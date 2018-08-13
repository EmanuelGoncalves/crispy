#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.plotting import get_palette_continuous


def tissue_coverage(samples):
    # Samplesheet
    ss = mp.get_samplesheet()

    # Data-frame
    plot_df = ss.loc[samples, 'Cancer Type']\
        .value_counts(ascending=True)\
        .reset_index()\
        .rename(columns={'index': 'Cancer type', 'Cancer Type': 'Counts'})\
        .sort_values('Counts', ascending=False)
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

    # Plot
    sns.barplot(plot_df['Counts'], plot_df['Cancer type'], color=cy.PAL_DBGD[0])

    for xx, yy in plot_df[['Counts', 'ypos']].values:
        plt.text(xx - .25, yy, str(xx), color='white', ha='right', va='center', fontsize=6)

    plt.grid(color=cy.PAL_DBGD[2], ls='-', lw=.3, axis='x')

    plt.xlabel('Number cell lines')
    plt.title('CRISPR-Cas9 screens\n({} cell lines)'.format(plot_df['Counts'].sum()))


def aucs_curves(df, geneset):
    pal = get_palette_continuous(df.shape[1])

    ax, auc_stats = cy.plot_cumsum_auc(df, geneset, plot_mean=False, legend=False, palette=pal)

    mean_auc = pd.Series(auc_stats['auc']).mean()

    plt.title('CRISPR-Cas9 {} fold-changes\nmean AURC = {:.2f}'.format(df.columns.name, mean_auc))

    return auc_stats


def aucs_scatter(x, y, data):
    ax = plt.gca()

    ax.scatter(data[x], data[y], c=cy.PAL_DBGD[0], s=4, marker='x', lw=0.5)
    sns.rugplot(data[x], height=.02, axis='x', c=cy.PAL_DBGD[0], lw=.3, ax=ax)
    sns.rugplot(data[y], height=.02, axis='y', c=cy.PAL_DBGD[0], lw=.3, ax=ax)

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, 'k--', lw=.3, zorder=0)

    ax.set_xlabel('{} fold-changes AURCs'.format(x.capitalize()))
    ax.set_ylabel('{} fold-changes AURCs'.format(y.capitalize()))

    return ax


if __name__ == '__main__':
    # - Imports
    # Gene-sets
    essential = pd.Series(list(mp.get_essential_genes())).rename('essential')
    nessential = pd.Series(list(mp.get_non_essential_genes())).rename('non-essential')

    # Non-expressed genes
    nexp = mp.get_non_exp()

    # Copy-number
    cnv = mp.get_copynumber()

    # CRISPR
    crispr = mp.get_crispr(mp.CRISPR_GENE_FC)
    crispr.columns.name = 'original'

    # Corrected CRISPR
    crispy = mp.get_crispr(mp.CRISPR_GENE_CORRECTED_FC)
    crispy.columns.name = 'corrected'

    # - Overlap samples
    samples = list(set(cnv).intersection(crispr).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Cancer type coverage
    tissue_coverage(samples)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/cancer_types_histogram.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Calculate AUCs of essential/non-essential genes in original/corrected fold-changes
    aucs = []
    for df in [crispr, crispy]:
        for gset in [essential, nessential]:
            df_aucs = aucs_curves(df[samples], gset)

            plt.gcf().set_size_inches(3, 3)
            plt.savefig('reports/qc_aucs_{}_{}.png'.format(df.columns.name, gset.name), bbox_inches='tight', dpi=600)
            plt.close('all')

            aucs.append(pd.DataFrame(dict(aurc=df_aucs['auc'], type=df.columns.name, geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})

    # - Scatter comparison
    for gset in [essential, nessential]:
        plot_df = pd.pivot_table(
            aucs.query("geneset == '{}'".format(gset.name)),
            index='sample', columns='type', values='aurc'
        )

        ax = aucs_scatter('original', 'corrected', plot_df)

        ax.set_title('AURCs {} genes'.format(gset.name))

        plt.gcf().set_size_inches(3, 3)
        plt.savefig('reports/qc_aucs_{}.png'.format(gset.name), bbox_inches='tight', dpi=600)
        plt.close('all')

    # - Export
    aucs.to_csv('data/qc_aucs.csv', index=False)
