#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.utils import bin_cnv
from scripts.plotting import plot_chromosome_seg, plot_rearrangements_seg


if __name__ == '__main__':
    # - Import
    # CRISPR library
    lib = cy.get_crispr_lib().groupby('gene')['chr'].first()

    # Non-expressed genes
    nexp = mp.get_non_exp()

    # Copy-number
    cnv = mp.get_copynumber()

    # Ploidy
    ploidy = mp.get_ploidy()

    # Chromosome copies
    chrm = mp.get_chr_copies()

    # Copy-number ratios
    cnv_ratios = mp.get_copynumber_ratios()

    # CRISPR
    crispr = mp.get_crispr(mp.CRISPR_GENE_FC)

    # Pancancer core essential
    pancan_core = cy.get_adam_core_essential()

    # - Overlap samples
    samples = list(set(cnv).intersection(crispr).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Assemble data-frame of non-expressed genes
    # Non-expressed genes CRISPR bias from copy-number
    df = pd.DataFrame({c: crispr[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples})

    # Concat copy-number information
    df = pd.concat([df[samples].unstack().rename('fc'), cnv[samples].unstack().rename('cnv'), cnv_ratios[samples].unstack().rename('ratio')], axis=1).dropna()
    df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

    # Append chromossome copy-number information
    df = df.assign(chr=lib[df['gene']].values)
    df = df.assign(chr_cnv=chrm.unstack().loc[list(map(tuple, df[['sample', 'chr']].values))].values)
    df = df.assign(ploidy=ploidy[df['sample']].values)

    # Bin copy-number profiles
    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cnv_bin=df['cnv'].apply(lambda v: bin_cnv(v, thresold=10)))
    df = df.assign(chr_cnv_bin=df['chr_cnv'].apply(lambda v: bin_cnv(v, thresold=6)))
    df = df.assign(ploidy_bin=df['ploidy'].apply(lambda v: bin_cnv(v, thresold=5)))

    # Filter
    df = df[~df['gene'].isin(pancan_core)]

    # -
    df[df['sample'].isin(mp.BRCA_SAMPLES)]\
        .query("ratio_bin == '2'")\
        .sort_values(['cnv', 'fc'], ascending=[False, False])\
        .tail(60)

    # -
    idx = 280956
    sample, chrm, genes = df.loc[idx, 'sample'], df.loc[idx, 'chr'], [df.loc[idx, 'gene']]
    # sample, chrm, genes = 'OV-90', 'chr22', ['KRTAP4-2']

    #
    ax, plot_df = plot_chromosome_seg(sample, chrm, genes_highlight=genes)
    plt.gcf().set_size_inches(8, 2)
    plt.savefig('reports/chromosomeplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    plot_rearrangements_seg(sample, chrm, genes_highlight=genes)
    plt.suptitle('{}'.format(sample), fontsize=10)
    plt.gcf().set_size_inches(8, 3)
    plt.savefig('reports/chromosomeplot_svs.png', bbox_inches='tight', dpi=600)
    plt.close('all')
