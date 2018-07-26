#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import PAL_DBGD
from natsort import natsorted
from crispy.utils import bin_cnv
from sklearn.metrics import roc_curve, auc
from scripts.plotting.copy_number_bias import copy_number_bias_aucs


if __name__ == '__main__':
    # - Import
    # CRISPR library
    lib = pd.read_csv(mp.LIBRARY, index_col=0).groupby('gene')['chr'].first()

    # Non-expressed genes
    nexp = pd.read_csv(mp.NON_EXP, index_col=0)

    # Copy-number
    cnv = pd.read_csv(mp.CN_GENE.format('snp'), index_col=0)

    # Chromosome copies
    chrm = pd.read_csv(mp.CN_CHR.format('snp'), index_col=0)

    # Copy-number ratios
    cnv_ratios = pd.read_csv(mp.CN_GENE_RATIO.format('snp'), index_col=0)

    # CRISPR
    c_gdsc_fc = pd.read_csv(mp.CRISPR_GENE_FC, index_col=0)

    # Samplesheet
    ss = pd.read_csv(mp.SAMPLESHEET, index_col=0)

    # Ploidy
    ploidy = pd.read_csv(mp.CN_PLOIDY.format('snp'), index_col=0, names=['sample', 'ploidy'])['ploidy']

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # -
    nx_genes = nexp[samples]
    nx_genes = list(nx_genes[nx_genes.sum(1) >= len(samples) * 1.].index)
    nx_genes = list(set.intersection(set(nx_genes), set(c_gdsc_fc.index), set(cnv_ratios.loc[(cnv_ratios[samples] >= 2).sum(1) >= 3].index)))

    #
    y = c_gdsc_fc.loc[nx_genes, samples]
    x = cnv_ratios.loc[nx_genes, samples]
    z = nexp.loc[nx_genes, samples]

    y[z != 1] = np.nan

    nx_genes_corr = x.corrwith(y, axis=1).sort_values()

    #
    y = c_gdsc_fc.loc[nx_genes, samples]
    x = cnv.loc[nx_genes, samples]
    z = nexp.loc[nx_genes, samples]

    y[z != 1] = np.nan

    nx_genes_corr_cnv = x.corrwith(y, axis=1).sort_values()

    # -
    corr_diff = (nx_genes_corr - nx_genes_corr_cnv).sort_values()


    # -
    g = 'LCE3B'
    # g = 'IQCF3'

    x = '{} copy-number ratio'.format(g)
    y = '{} CRISPR fold-change'.format(g)

    plot_df = pd.concat([
        cnv_ratios.loc[g, samples].rename(x),
        c_gdsc_fc.loc[g, samples].rename(y),
        ss.loc[samples, ['Cancer Type']],
        cnv.loc[g, samples].rename('cnv'),
        ploidy.loc[samples],
        nexp.loc[g, samples].rename('nexp')
    ], axis=1).query('nexp == 1').sort_values(x)

    sns.jointplot(x, y, data=plot_df, color=PAL_DBGD[0], edgecolor='white')
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/ratio_validataion.png', bbox_inches='tight', dpi=600)
    plt.close('all')


