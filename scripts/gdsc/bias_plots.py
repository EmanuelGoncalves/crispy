#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted


def tissue_coverage(samples):
    # Samplesheet
    ss = gdsc.get_sample_info()

    # Data-frame
    plot_df = ss.loc[samples, 'Cancer Type']\
        .value_counts(ascending=True)\
        .reset_index()\
        .rename(columns={'index': 'Cancer type', 'Cancer Type': 'Counts'})\
        .sort_values('Counts', ascending=False)
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

    # Plot
    sns.barplot(plot_df['Counts'], plot_df['Cancer type'], color=cy.QCplot.PAL_DBGD[0])

    for xx, yy in plot_df[['Counts', 'ypos']].values:
        plt.text(xx - .25, yy, str(xx), color='white', ha='right', va='center', fontsize=6)

    plt.grid(color=cy.QCplot.PAL_DBGD[2], ls='-', lw=.3, axis='x')

    plt.xlabel('Number cell lines')
    plt.title('CRISPR-Cas9 screens\n({} cell lines)'.format(plot_df['Counts'].sum()))


if __name__ == '__main__':
    # - Non-expressed genes
    nexp = gdsc.get_non_exp()

    # - Crispy bed files
    beds = gdsc.import_crispy_beds()

    # - Aggregate by gene
    agg_fun = dict(fold_change=np.mean, copy_number=np.mean, ploidy=np.mean, chr_copy=np.mean)

    bed_nexp_gene = []

    for s in beds:
        df = beds[s].groupby('gene').agg(agg_fun)
        df = df.assign(scaled=cy.Crispy.scale_crispr(df['fold_change']))
        df = df.reindex(nexp[nexp[s] == 1].index).dropna()
        df = df.assign(sample=s)

        bed_nexp_gene.append(df)

    bed_nexp_gene = pd.concat(bed_nexp_gene).reset_index()
    bed_nexp_gene = bed_nexp_gene.assign(copy_number_bin=bed_nexp_gene['copy_number'].apply(cy.Utils.bin_cnv, args=(10,)))

    # - Get ploidy
    ploidy = pd.Series({s: beds[s]['ploidy'].mean() for s in beds})

    # - Cancer type coverage
    tissue_coverage(list(beds))

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/cancer_types_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - AURC curves
    order = natsorted(set(bed_nexp_gene['copy_number_bin']))[2:]

    cn_curves = {c: cy.QCplot.recall_curve(bed_nexp_gene['scaled'], bed_nexp_gene.query('copy_number_bin == @c').index) for c in order}

    # Plot
    palette = list(reversed(cy.QCplot.get_palette_continuous(len(cn_curves))))

    for i, c in enumerate(reversed(order)):
        plt.plot(cn_curves[c][0], cn_curves[c][1], label=f'{c}: AURC={cn_curves[c][2]:.2f}', lw=1., c=palette[i], alpha=.8)

    plt.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel('Ranked genes by fold-change')
    plt.ylabel(f'Recall')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.legend(frameon=False, title='Copy-number', prop={'size': 6}, fontsize=6)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/gdsc_bias_copynumber_curves.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Per sample
    cn_aucs_df = pd.concat([cy.QCplot.recall_curve_discretise(
        bed_nexp_gene.query('sample == @s')['scaled'],
        bed_nexp_gene.query('sample == @s')['copy_number'],
        10
    ).assign(sample=s) for s in beds])

    # Add ploidy
    cn_aucs_df = cn_aucs_df.assign(ploidy=ploidy[cn_aucs_df['sample']].values)
    cn_aucs_df = cn_aucs_df.assign(ploidy_bin=cn_aucs_df['ploidy'].apply(cy.Utils.bin_cnv, args=(5,)))

    # Plot bias per cell line
    ax = cy.QCplot.bias_boxplot(
        cn_aucs_df[~cn_aucs_df['copy_number'].isin(['0', '1'])], x='copy_number', notch=True, add_n=True, n_text_y=.05
    )

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/gdsc_bias_copynumber_boxplots.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Plot bias per cell line group by ploidy
    order = natsorted(set(cn_aucs_df['ploidy_bin']))
    palette = dict(zip(*(order, cy.QCplot.get_palette_continuous(len(order)))))

    ax = cy.QCplot.bias_boxplot(
        cn_aucs_df[~cn_aucs_df['copy_number'].isin(['0', '1'])], x='copy_number', hue='ploidy_bin', notch=False, add_n=False, n_text_y=.05, palette=palette
    )

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.legend(frameon=False, title='Ploidy', prop={'size': 6}, fontsize=6)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/gdsc_bias_copynumber_boxplots_by_ploidy.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
