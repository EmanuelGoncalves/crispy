#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy import bipal_dbgd
from crispy.utils import bin_cnv
from sklearn.metrics import roc_curve, auc
from scripts.plotting.qc_bias_assessment import auc_curves


def ratios_arocs(y_true, y_pred, data, outfile, exclude_labels=set()):
    order = natsorted(set(data[y_true]).difference(exclude_labels))

    pal = sns.light_palette(bipal_dbgd[0], n_colors=len(order) + 1).as_hex()[1:]

    ax = plt.gca()

    for c, thres in zip(pal, order):
        fpr, tpr, _ = roc_curve((data[y_true] == thres).astype(int), -data[y_pred])
        ax.plot(fpr, tpr, label='%s: AUC=%.2f' % (thres, auc(fpr, tpr)), lw=1.5, c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Copy-number ratio impact in shRNA\n(non-expressed genes)')
    legend = ax.legend(loc=4, title='Copy-number ratio', prop={'size': 10})
    legend.get_title().set_fontsize('10')

    plt.gcf().set_size_inches(3.5, 3.5)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


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
    ratios_arocs('ratio_bin', 'shrna', df, 'reports/crispy/copynumber_ratio_shrna.png')
