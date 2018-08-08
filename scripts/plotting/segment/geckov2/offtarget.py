#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.plotting.segment.geckov2.calculate_fold_change import GECKOV2_SGRNA_MAP


def bin_bkdist(distance):
    if distance == -1:
        bin_distance = 'diff. chr.'

    elif 1 < distance < 10e3:
        bin_distance = '1-10 kb'

    elif 10e3 < distance < 100e3:
        bin_distance = '1-100 kb'

    elif 100e3 < distance < 1e6:
        bin_distance = '0.1-1 Mb'

    elif 1e6 < distance < 10e6:
        bin_distance = '1-10 Mb'

    else:
        bin_distance = '>10 Mb'

    return bin_distance


def lib_off_targets():
    lib = pd.read_csv(GECKOV2_SGRNA_MAP, sep='\t').dropna(subset=['Chromosome', 'Pos', 'Strand'])

    lib = lib.rename(columns={'Chromosome': '#chr', 'Pos': 'start'})

    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])
    lib['gene'] = lib['Guide'].apply(lambda v: v.split('_')[1])
    lib['start'] = lib['start'].astype(int)
    lib['end'] = lib['start'] + lib['sgrna'].apply(len) + lib['PAM'].apply(len)

    sgrna_maps = lib['sgrna'].value_counts().rename('ncuts').to_frame()
    sgrna_maps = sgrna_maps[sgrna_maps['ncuts'] > 1]

    sgrna_nchrs = lib.groupby('sgrna')['#chr'].agg(lambda x: len(set(x)))
    sgrna_maps = sgrna_maps.assign(nchrs=sgrna_nchrs.loc[sgrna_maps.index].values)

    sgrna_genes = lib.groupby('sgrna')['gene'].agg(lambda x: ';'.join(set(x)))
    sgrna_maps = sgrna_maps.assign(genes=sgrna_genes.loc[sgrna_maps.index].values)
    sgrna_maps = sgrna_maps.assign(ngenes=sgrna_maps['genes'].apply(lambda x: len(x.split(';'))))

    sgrna_distance = lib.groupby('sgrna')['start'].agg([np.min, np.max]).eval('amax-amin')
    sgrna_maps = sgrna_maps.assign(distance=[sgrna_distance[g] if sgrna_nchrs[g] == 1 else -1 for g in sgrna_maps.index])
    sgrna_maps = sgrna_maps.assign(distance_bin=sgrna_maps['distance'].apply(bin_bkdist).values)

    return sgrna_maps


def scale_geckov2(fc):
    lib = pd.read_csv(GECKOV2_SGRNA_MAP, sep='\t')
    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])
    lib['gene'] = lib['Guide'].apply(lambda v: v.split('_')[1])

    ess = cy.get_essential_genes()
    ness = cy.get_non_essential_genes()

    ess_metric = np.median(fc.reindex(lib[lib['gene'].isin(ess)]['sgrna']).dropna(), axis=0)
    ness_metric = np.median(fc.reindex(lib[lib['gene'].isin(ness)]['sgrna']).dropna(), axis=0)

    return fc.subtract(ness_metric).divide(ness_metric - ess_metric)


if __name__ == '__main__':
    # -
    fc = pd.read_csv('data/geckov2/sgrna_fc.csv', index_col=0)
    fc = scale_geckov2(fc)

    # -
    sgrna_maps = lib_off_targets()

    # -
    plot_df = pd.concat([sgrna_maps.query('ncuts <= 6'), fc], axis=1, sort=False).dropna().reset_index()
    plot_df = pd.melt(plot_df, id_vars=['index'] + list(sgrna_maps))
    plot_df = plot_df.query('distance > 0').query('ngenes == 1')

    #
    hue_order = ['1-10 kb', '1-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb']

    for s in list(fc):
        sns.boxplot('ncuts', 'value', 'distance_bin', data=plot_df.query('variable == @s'), notch=True, hue_order=hue_order, linewidth=.3, fliersize=1)

        plt.legend(title='', prop={'size': 3})

        plt.axhline(-1, lw=.3, ls='--')

        plt.gcf().set_size_inches(2, 2)
        plt.savefig(f'reports/geckov2_off_target_{s}.png', bbox_inches='tight', dpi=600)
        plt.close('all')

    sns.boxplot('ncuts', 'value', 'distance_bin', data=plot_df, notch=True, hue_order=hue_order, linewidth=.3, fliersize=1)

    plt.legend(title='', prop={'size': 3}, frameon=False)

    plt.axhline(-1, lw=.3, ls='--')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/geckov2_off_target.png', bbox_inches='tight', dpi=600)
    plt.close('all')