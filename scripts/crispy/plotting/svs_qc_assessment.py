#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from sklearn.metrics import roc_auc_score
from crispy.benchmark_plot import plot_cumsum_auc
from crispy.utils import multilabel_roc_auc_score_array, bin_cnv


if __name__ == '__main__':

    svs_dir = 'data/crispy/gdsc_brass/'

    brca_samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    essential = pd.read_csv('data/gene_sets/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

    # -
    svs_crispy = {}
    for s in brca_samples:
        svs_crispy[s] = {
            'copynumber': pd.read_csv('{}/{}.copynumber.csv'.format(svs_dir, s), index_col=0),
            'sv': pd.read_csv('{}/{}.svs.csv'.format(svs_dir, s), index_col=0)
        }

    # -
    cnv_bias = pd.DataFrame([
        {'sample': s, 'type': k, 'cnv': c, 'auc': v}

        for s in svs_crispy
        for k, df in svs_crispy[s].items()
        for c, v in multilabel_roc_auc_score_array(df['cnv'].apply(lambda v: bin_cnv(v, 10)), df['regressed_out'], min_events=5).items()
    ])

    sns.boxplot('cnv', 'auc', 'type', data=cnv_bias, order=natsorted(set(cnv_bias['cnv'])), fliersize=2)
    plt.show()

    # -
    ess_aucs = pd.DataFrame({
        s: {k: plot_cumsum_auc(df[['regressed_out']], essential, plot_mean=True, legend=False)[1]['auc']['regressed_out'] for k, df in svs_crispy[s].items()} for s in svs_crispy
    }).T
    plt.close()

    sns.jointplot(ess_aucs['copynumber'], ess_aucs['sv'], kind='scatter', color=bipal_dbgd[0])
    plt.show()
