#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scripts as mp
import matplotlib.pyplot as plt
from scripts.plotting.wgs.svplot import import_brass_bedpe, plot_rearrangements


if __name__ == '__main__':
    svs_dir = '{}/{}.{}.csv'

    # - Import CRISPR lib
    crispr_lib = pd.read_csv(mp.LIBRARY, index_col=0).groupby('gene').agg(
        {'start': np.min, 'end': np.max}
    )

    # - Import Crispy results
    svs_types = ['tandemduplication', 'deletion', 'inversion']
    fit_type = {'svs': 'SV', 'cnv': 'Copy-number', 'all': 'Both'}

    svs_crispy = {
        t: {
            s: pd.read_csv(svs_dir.format(mp.CRISPY_WGS_OUTDIR, s, t), index_col=0).rename(columns={'fit_by': 'chr'}) for s in mp.BRCA_SAMPLES
        } for t in fit_type
    }

    # -
    sample, chrm, t = 'HCC38', 'chr5', 'all'

    # for sample, chrm in order.index:
    for sample in svs_crispy['all']:
        for chrm in set(svs_crispy['all']['HCC38']['chr']):
            print('{} {}'.format(sample, chrm))

            t = 'all'

            # Import BRASS bedpe
            bedpe = import_brass_bedpe('{}/{}.brass.annot.bedpe'.format(mp.WGS_BRASS_BEDPE, sample), bkdist=-1, splitreads=True)

            # Import WGS sequencing depth
            ngsc = pd.read_csv(
                'data/gdsc/wgs/brass_intermediates/{}.ngscn.abs_cn.bg'.format(sample), sep='\t', header=None,
                names=['chr', 'start', 'end', 'absolute_cn', 'norm_logratio'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
            )
            ngsc = ngsc.assign(chr=ngsc['chr'].apply(lambda x: 'chr{}'.format(x)).values)

            # Concat CRISPR measurements
            crispr = pd.concat([svs_crispy[t][sample], crispr_lib], axis=1).dropna()

            # Plot
            xlim = (0e6, 100e6)

            ax1, ax2, ax3 = plot_rearrangements(bedpe, crispr, ngsc, chrm, winsize=1e4, show_legend=True, xlim=xlim)

            plt.suptitle('{}'.format(sample), fontsize=10)

            plt.gcf().set_size_inches(7, 2)
            plt.savefig('reports/crispy/brass_svplot_{}_{}_{}.png'.format(sample, chrm, t), bbox_inches='tight', dpi=600)
            plt.close('all')
