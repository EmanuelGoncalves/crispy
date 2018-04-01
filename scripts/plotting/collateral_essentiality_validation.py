#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd


if __name__ == '__main__':
    # - cBioPortal
    # Import
    df = pd.read_csv('data/cbioportal_neurod2_cna.tsv', sep='\t')
    df = df.assign(amp=(df['Alteration'] == 'CNA: AMP;').astype(int))

    # Build data-frame
    plot_df = pd.concat([
        df.groupby('Study')['amp'].sum().rename('Amplifications'), df.groupby('Study')['Sample'].count().rename('Total')
    ], axis=1)
    plot_df = plot_df.assign(Percentage=(plot_df.eval('Amplifications / Total') * 100).round(2)).sort_values('Percentage')
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

    # Plot
    plt.barh(plot_df['ypos'], plot_df['Percentage'], .8, color=bipal_dbgd[0], align='center')

    for xx, yy, l in plot_df[['ypos', 'Percentage', 'Total']].values:
        if yy > 0:
         plt.text(yy - .05, xx, '{:.0f}%'.format(yy), color='white', ha='right', va='center', fontsize=6)

    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], [i.replace('_', ' ').upper() for i in plot_df.index])

    plt.xlabel('% of samples with NEUROD2 amplifications')
    plt.title('cBioPortal BRCA samples cohorts')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/crispy/cnv_cbioportal_NEUROD2.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - GTex
    gtex_tpm = pd.read_csv('data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct', sep='\t', index_col=1).drop(['gene_id'], axis=1)

    plot_df = gtex_tpm.loc['NEUROD2', :].reset_index()
    plot_df.columns = ['Tissue', 'Expression']
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))
    plot_df = plot_df.assign(tissue=[i.split(' - ')[0] for i in plot_df['Tissue']])

    # Plot
    plt.bar(plot_df['ypos'], plot_df['Expression'], .8, color=bipal_dbgd[0], align='center')

    # for xx, yy, l in plot_df[['ypos', 'Expression', 'Tissue']].values:
    #     if yy > 0:
    #      plt.text(yy - .05, xx, '{:.2f}%'.format(yy), color='white', ha='right', va='center', fontsize=6, rotation=90)

    plt.xticks(plot_df['ypos'])
    plt.xticks(plot_df['ypos'], plot_df['Tissue'], rotation=90)

    plt.ylabel('NEUROD2 gene median TPM')
    plt.title('GTex RNA-seq gene expression')

    plt.gcf().set_size_inches(10, 3)
    plt.savefig('reports/crispy/collateral_essentiality_gtex_NEUROD2.png', bbox_inches='tight', dpi=600)
    plt.close('all')
