#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.utils import bin_cnv
from scripts.plotting.qc_bias_assessment import auc_curves
from scripts.plotting.qc_bias_assessment import copy_number_bias_aucs


if __name__ == '__main__':
    # - Imports
    # (non)essential genes
    essential = pd.read_csv(mp.HART_ESSENTIAL, sep='\t')['gene'].rename('essential')

    # CRISPR lib
    lib = pd.read_csv(mp.LIBRARY, index_col=0)
    lib = lib.assign(pos=lib[['start', 'end']].mean(1).values)

    # Samplesheet
    ss = pd.read_csv(mp.SAMPLESHEET).dropna(subset=['CCLE ID']).set_index('CCLE ID')

    # DRIVE shRNA RSA scores
    shrna = pd.read_csv('data/ccle/DRIVE_RSA_data.txt', sep='\t', index_col=0)
    shrna.columns = [c.upper() for c in shrna]
    shrna = shrna.rename(columns=ss['Cell Line Name'].to_dict())

    # Copy-number ratios
    cnv_ratios = pd.read_csv(mp.CN_GENE_RATIO.format('snp'), index_col=0)

    # Non-expressed genes
    nexp = pd.read_csv(mp.NON_EXP, index_col=0)

    # - Overlap
    samples = list(set(shrna).intersection(cnv_ratios).intersection(nexp))
    print('Samples: {}'.format(len(samples)))

    # - Recapitulated essential genes with shRNA
    auc_stats_fc_ess = auc_curves(shrna[samples], essential, 'reports/crispy/qc_aucs_fc_essential_shrna.png')

    # - Assemble data-frame of non-expressed genes
    # Non-expressed genes CRISPR bias from copy-number
    df = pd.DataFrame({c: shrna[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples})

    # Concat copy-number information
    df = pd.concat([
        df[samples].unstack().rename('shrna'),
        cnv_ratios[samples].unstack().rename('ratio')
    ], axis=1).dropna()
    df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

    # Bin copy-number profiles
    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))

    # - Plot: Copy-number ratio vs CRISPR bias
    copy_number_bias_aucs(
        df, rank_label='shrna', thres_label='ratio_bin',
        outfile='reports/crispy/copynumber_ratio_shrna.png',
        title='Copy-number effect on shRNA\n(non-expressed genes)'
    )
