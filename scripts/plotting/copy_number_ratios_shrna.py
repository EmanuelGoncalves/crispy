#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.utils import bin_cnv
from scripts.plotting import bias_arocs


if __name__ == '__main__':
    # - Imports
    # Non-expressed genes
    nexp = mp.get_non_exp()

    # Copy-number
    cnv = mp.get_copynumber()

    # Copy-number ratios
    cnv_ratios = mp.get_copynumber_ratios()

    # shRNA
    shrna = mp.get_shrna()

    # - Overlap
    samples = list(set(shrna).intersection(cnv_ratios).intersection(nexp))
    print('Samples: {}'.format(len(samples)))

    # - Assemble data-frame of non-expressed genes
    # Non-expressed genes CRISPR bias from copy-number
    df = pd.DataFrame({c: shrna[c].reindex(nexp.loc[nexp[c] == 1, c].index) for c in samples})

    # Concat copy-number information
    df = pd.concat([df[samples].unstack().rename('fc'), cnv[samples].unstack().rename('cnv'), cnv_ratios[samples].unstack().rename('ratio')], axis=1).dropna()
    df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

    # Bin copy-number profiles
    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cnv_bin=df['cnv'].apply(lambda v: bin_cnv(v, thresold=10)))

    # - Plot
    # Overall copy-number ratio bias AUCs
    ax, _ = bias_arocs(df, groupby='ratio_bin')
    ax.set_title('Copy-number ratio effect on shRNA\n(non-expressed genes)')
    ax.legend(loc=4, title='Copy-number ratio', prop={'size': 6}, frameon=False).get_title().set_fontsize(6)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/ratios_aucs_shrna.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Overall copy-number ratio bias AUCs
    ax, _ = bias_arocs(df, groupby='cnv_bin')
    ax.set_title('Copy-number effect on shRNA\n(non-expressed genes)')
    ax.legend(loc=4, title='Copy-number', prop={'size': 6}, frameon=False).get_title().set_fontsize(6)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/original_bias_copynumber_aucs_shrna.png', bbox_inches='tight', dpi=600)
    plt.close('all')
