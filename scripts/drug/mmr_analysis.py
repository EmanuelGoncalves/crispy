#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scripts.drug as dc
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap


if __name__ == '__main__':
    # - Import
    # Samplesheet
    ss = pd.read_csv(dc.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # WES
    wes = pd.read_csv('data/gdsc/WES_variants.csv')

    # Mutation load
    n_mutations = wes.groupby('SAMPLE')['Classification'].count().rename('mutations')

    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv(dc.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    crispr_scaled = dc.scale_crispr(crispr)

    # Methylation
    methy = pd.read_csv('data/gdsc/methylation/methy_beta_gene_promoter.csv', index_col=0)

    # MOBEMS
    mobem = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.all.csv', index_col=0)

    # -
    samples = list(set(crispr).intersection(n_mutations.index).intersection(ss.index).intersection(methy))
    print('Samples: %d' % len(samples))

    # -
    plot_df = pd.concat([
        n_mutations[samples],

        ss.loc[samples, ['Microsatellite', 'Cancer Type']],

        ss.loc[samples, 'Cancer Type'].apply(lambda x: int(x == 'Colorectal Carcinoma')).rename('Colorectal Carcinoma'),

        ss.loc[samples, 'Microsatellite'].apply(lambda x: int(x == 'MSI-H')).rename('MSI'),

        crispr_scaled.loc['WRN'].apply(lambda x: int(x < -1)).rename('WRN (essential)'),

        methy.loc['MLH1', samples].apply(lambda x: int(x > .66)).rename('MLH1 (hypermethylation)'),

        wes[wes['Gene'].isin(['POLE'])].drop_duplicates(subset=['SAMPLE', 'Gene']).set_index('SAMPLE').assign(value=1)['value'].reindex(samples).replace(np.nan, 0).rename('POLE (mutation)')

    ], axis=1).dropna()
    plot_df = plot_df.reset_index().sort_values('mutations', ascending=False)
    plot_df = plot_df.assign(pos=range(plot_df.shape[0]))

    #
    pal = sns.light_palette(bipal_dbgd[0], n_colors=2)

    #
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [4, 1]})
    plt.subplots_adjust(wspace=.1, hspace=.1)

    #
    ax1.bar(plot_df['pos'], plot_df['mutations'], color=bipal_dbgd[0], align='edge')

    ax1.yaxis.grid(True, color=pal[0], linestyle='-', linewidth=.3)
    ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    ax1.set_xlabel('')
    ax1.set_ylabel('# mutations')

    ax1.set_adjustable('box-forced')

    #
    cmap = ListedColormap(pal)

    sns.heatmap(plot_df.set_index('pos')[['Colorectal Carcinoma', 'MSI', 'WRN (essential)', 'POLE (mutation)', 'MLH1 (hypermethylation)']].T, cbar=False, ax=ax2, cmap=cmap, lw=.3)

    ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    ax2.set_adjustable('box-forced')

    ax2.set_xlabel('')

    #
    plt.suptitle('Genomic landscape of WRN dependence')

    #
    plt.gcf().set_size_inches(8, 4)
    plt.savefig('reports/drug/mmr_mutation_count.png', bbox_inches='tight', dpi=600)
    plt.close('all')
