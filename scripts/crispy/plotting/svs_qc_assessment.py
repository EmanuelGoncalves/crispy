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
from scripts.crispy.plotting.svplot import import_brass_bedpe, plot_rearrangements


if __name__ == '__main__':
    svs_dir = 'data/crispy/gdsc_brass/'

    crispr_lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0).groupby('gene').agg(
        {'start': np.min, 'end': np.max}
    )

    brca_samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    essential = pd.read_csv('data/gene_sets/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

    # -
    svs_crispy = {}
    for s in brca_samples:
        svs_crispy[s] = {
            'copynumber': pd.read_csv('{}/{}.copynumber.csv'.format(svs_dir, s), index_col=0).rename(columns={'fit_by': 'chr'}),
            'sv': pd.read_csv('{}/{}.svs.csv'.format(svs_dir, s), index_col=0).rename(columns={'fit_by': 'chr'}),
            'all': pd.read_csv('{}/{}.all.csv'.format(svs_dir, s), index_col=0).rename(columns={'fit_by': 'chr'})
        }

    # -
    diff_var_exp = pd.DataFrame({c: svs_crispy[c]['copynumber']['var_exp'] - svs_crispy[c]['sv']['var_exp'] for c in svs_crispy})

    # -
    sample, chrm, = 'HCC1937', 'chr18'
    for sample in brca_samples:
        #
        bedpe = import_brass_bedpe('data/gdsc/wgs/brass_bedpe/{}.brass.annot.bedpe'.format(sample), bkdist=None, splitreads=True)

        #
        ngsc = pd.read_csv(
            'data/gdsc/wgs/brass_intermediates/{}.ngscn.abs_cn.bg'.format(sample), sep='\t', header=None,
            names=['chr', 'start', 'end', 'absolute_cn', 'norm_logratio'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
        )
        ngsc = ngsc.assign(chr=ngsc['chr'].apply(lambda x: 'chr{}'.format(x)).values)

        #
        crispr = pd.concat([svs_crispy[sample]['sv'], crispr_lib], axis=1).dropna()

        for chrm in set(bedpe['chr1']).union(bedpe['chr2']):
            print('{} {}'.format(sample, chrm))

            #
            ax1, ax2, ax3 = plot_rearrangements(bedpe, crispr, ngsc, chrm)

            plt.suptitle('{}'.format(sample))

            #
            plt.gcf().set_size_inches(8, 3)
            plt.savefig('reports/crispy/svplots/svplot_{}_{}.pdf'.format(sample, chrm), bbox_inches='tight')
            plt.close('all')
