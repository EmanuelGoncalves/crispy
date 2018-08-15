#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
from natsort import natsorted
import matplotlib.pyplot as plt
from crispy.utils import bin_cnv
from plotting import get_palette_continuous, FLIERPROPS, MEANLINEPROPS
from plotting.geckov2.fit_gp import GECKOV2_SGRNA_MAP


def import_seg_crispr_beds():
    # Samples
    samples = list(pd.read_csv('data/geckov2/Achilles_v3.3.8.samplesheet.txt', sep='\t', index_col=1)['name'])
    samples = [s for s in samples if os.path.exists(f'data/geckov2/crispr_bed/{s}.crispr.snp.bed')]

    # SNP6 + CRISPR BED files
    beds = {s: pd.read_csv(f'data/geckov2/crispr_bed/{s}.crispr.snp.bed', sep='\t') for s in samples}

    return beds


def bin_bkdist(distance):
    if distance == -1:
        bin_distance = 'diff. chr.'

    elif distance == 0:
        bin_distance = '0'

    elif 1 < distance < 10e3:
        bin_distance = '1-10 kb'

    elif 10e3 < distance < 100e3:
        bin_distance = '10-100 kb'

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
    lib['cut'] = lib['#chr'] + '_' + lib['start'].astype(str)

    sgrna_maps = lib.groupby('sgrna')['cut'].agg(lambda x: len(set(x))).sort_values().rename('ncuts').reset_index()
    sgrna_maps = sgrna_maps[sgrna_maps['ncuts'] > 1]

    sgrna_nchrs = lib.groupby('sgrna')['#chr'].agg(lambda x: len(set(x)))
    sgrna_maps = sgrna_maps.assign(nchrs=sgrna_nchrs.loc[sgrna_maps['sgrna']].values)

    sgrna_genes = lib.groupby('sgrna')['gene'].agg(lambda x: ';'.join(set(x)))
    sgrna_maps = sgrna_maps.assign(genes=sgrna_genes.loc[sgrna_maps['sgrna']].values)
    sgrna_maps = sgrna_maps.assign(ngenes=sgrna_maps['genes'].apply(lambda x: len(x.split(';'))))

    sgrna_distance = lib.groupby('sgrna')['start'].agg([np.min, np.max]).eval('amax-amin')
    sgrna_maps = sgrna_maps.assign(distance=[sgrna_distance[g] if sgrna_nchrs[g] == 1 else -1 for g in sgrna_maps['sgrna']])
    sgrna_maps = sgrna_maps.assign(distance_bin=sgrna_maps['distance'].apply(bin_bkdist).values)

    sgrna_maps = sgrna_maps.set_index('sgrna')

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
    beds = import_seg_crispr_beds()

    # -
    ploidy = pd.Series({s: beds[s]['ploidy'].mean() for s in beds})

    # -
    fc = pd.read_csv('data/geckov2/sgrna_fc.csv', index_col=0)
    fc = scale_geckov2(fc)

    # -
    sgrna_maps = lib_off_targets()
    sgrna_maps['distance_kb'] = sgrna_maps['distance'] / 1e4
    sgrna_maps['ncuts_bin'] = sgrna_maps['ncuts'].apply(lambda v: bin_cnv(v, thresold=10))
    sgrna_maps['fc_mean'] = fc.loc[sgrna_maps.index].mean(1)

    # -
    plot_df = sgrna_maps.dropna().query('ngenes == 1')
    col_order = natsorted(set(plot_df['ncuts_bin']))

    g = sns.FacetGrid(plot_df, col='ncuts_bin', col_wrap=3, col_order=col_order, sharey=False, sharex=False)
    g.map(sns.regplot, 'distance_kb', 'fc_mean')
    # g.map(sns.kdeplot, 'distance_log', 'fc_mean', shade=True, n_levels=30, cmap='RdGy', shade_lowest=False)
    plt.savefig(f'reports/offtarget_distance.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    plot_df = sgrna_maps.query('ngenes == 1')
    plot_df = plot_df[(plot_df['nchrs'] == 1) | (plot_df['ncuts'] == plot_df['nchrs'])]

    order = natsorted(set(plot_df['ncuts_bin']))
    hue_order = ['1-10 kb', '10-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb', 'diff. chr.']
    sns.boxplot('ncuts_bin', 'fc_mean', 'distance_bin', data=plot_df, order=order, hue_order=hue_order, notch=True)
    plt.savefig(f'reports/offtarget_distance_boxplots.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    hue_order = ['1-10 kb', '10-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb', 'diff. chr.']

    plot_df = pd.concat([sgrna_maps.query('ncuts <= 10'), fc], axis=1, sort=False).dropna().reset_index()
    plot_df = pd.melt(plot_df, id_vars=['index'] + list(sgrna_maps))
    plot_df = plot_df[[ncuts == nchrs if dist == -1 else True for ncuts, nchrs, dist in plot_df[['ncuts', 'nchrs', 'distance']].values]]
    plot_df['ncuts'] = plot_df['ncuts'].astype(int)

    sns.boxplot(
        'ncuts', 'value', 'distance_bin', data=plot_df, notch=True, hue_order=hue_order, linewidth=.3,
        meanprops=MEANLINEPROPS, flierprops=FLIERPROPS, palette=get_palette_continuous(len(hue_order))
    )

    plt.legend(title='', prop={'size': 3}, frameon=False)

    plt.axhline(-1, lw=.1, ls='--', c='black', zorder=0)
    plt.axhline(0, lw=.1, ls='--', c='black', zorder=0)

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/geckov2_off_target.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    plot_df = pd.concat([
        sgrna_maps.query('ncuts <= 10 & distance == -1 & ncuts == nchrs'), fc
    ], axis=1, sort=False).dropna().reset_index()
    plot_df = pd.melt(plot_df, id_vars=['index'] + list(sgrna_maps))
    plot_df = plot_df.assign(ploidy=ploidy[plot_df['variable']].apply(lambda v: bin_cnv(v, 5)).values)
    plot_df['ncuts'] = plot_df['ncuts'].astype(int)

    hue_order = ['2', '3', '4', '5+']

    sns.boxplot(
        'ncuts', 'value', 'ploidy', data=plot_df, notch=True, hue_order=hue_order, linewidth=.3,
        meanprops=MEANLINEPROPS, flierprops=FLIERPROPS, palette=get_palette_continuous(len(hue_order))
    )

    plt.legend(title='', prop={'size': 3}, frameon=False)

    plt.axhline(-1, lw=.1, ls='--', c='black', zorder=0)
    plt.axhline(0, lw=.1, ls='--', c='black', zorder=0)

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/geckov2_off_target_ploidy.png', bbox_inches='tight', dpi=600)
    plt.close('all')
