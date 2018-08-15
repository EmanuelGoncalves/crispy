#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.utils import bin_cnv
from plotting import get_palette_continuous, FLIERPROPS, MEANLINEPROPS
from plotting.geckov2.offtarget import import_seg_crispr_beds, lib_off_targets


def ratios_heatmap_bias(df, x='ncuts_bin', y='cn_bin', z='fc_scaled', min_evetns=10):
    plot_df = df.copy()

    counts = {(g, c) for g, c, v in plot_df.groupby([x, y])[z].count().reset_index().values if v <= min_evetns}

    plot_df = plot_df[[(g, c) not in counts for g, c in plot_df[[x, y]].values]]
    plot_df = plot_df.groupby([x, y])[z].mean().reset_index()
    plot_df = pd.pivot_table(plot_df, index=x, columns=y, values=z)

    y_order = natsorted(plot_df.index, reverse=False)
    x_order = natsorted(plot_df.columns, reverse=True)

    g = sns.heatmap(
        plot_df.loc[y_order, x_order].T, cmap='RdGy_r', annot=True, fmt='.2f', square=True,
        linewidths=.3, cbar=False, annot_kws={'fontsize': 7}, center=0
    )
    plt.setp(g.get_yticklabels(), rotation=0)


if __name__ == '__main__':
    # Import libraries bed
    beds = import_seg_crispr_beds()

    # Off-targets
    sgrnas = lib_off_targets()

    # -
    df = []
    for s in beds:
        df_ = beds[s][beds[s]['sgrna_id'].isin(sgrnas.index)].assign(sample=s)

        df_ = df_.assign(ncuts=sgrnas.loc[df_['sgrna_id']]['ncuts'].values)
        df_ = df_.assign(ncuts_bin=df_['ncuts'].apply(lambda v: bin_cnv(v, 15)).values)

        df_ = df_.assign(nchrs=sgrnas.loc[df_['sgrna_id']]['nchrs'].values)
        df_ = df_.assign(nchrs_bin=df_['nchrs'].apply(lambda v: bin_cnv(v, 6)).values)

        df_ = df_.assign(cn_bin=df_['cn'].apply(lambda v: bin_cnv(v, 8)).values)
        df_ = df_.assign(ratio_bin=df_['ratio'].apply(lambda v: bin_cnv(v, 4)).values)
        df_ = df_.assign(ploidy_bin=df_['ploidy'].apply(lambda v: bin_cnv(v, 4)).values)

        df_ = df_.assign(distance=sgrnas.loc[df_['sgrna_id']]['distance'].values)
        df_ = df_.assign(distance_bin=sgrnas.loc[df_['sgrna_id']]['distance_bin'].values)

        df_ = df_.assign(seg_id=df_['chr'] + '_' + df_['start'].astype(str) + '_' + df_['end'].astype(str))

        sgrna_seg_count = df_.groupby('sgrna_id')['seg_id'].agg(lambda x: len(set(x)))
        df_ = df_[df_['sgrna_id'].isin(sgrna_seg_count[sgrna_seg_count == 1].index)]

        df.append(df_)

    df = pd.concat(df).reset_index(drop=True)

    df['same_chr'] = (df['distance'] != -1).astype(int)

    # - Copy-number ratio bias heatmap
    ratios_heatmap_bias(df.query('distance != -1').query('ncuts <= 6'), y='cn_bin')

    plt.xlabel('Guide number of cuts')
    plt.ylabel('Segment absolute copy-number')
    plt.title('Mean CRISPR-Cas9 fold-change')

    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/offtargets_seg_fc_heatmap.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    hue_order = natsorted(set(df['ratio_bin']))
    order = natsorted(set(df['ncuts_bin']))

    g = sns.catplot(
        'ncuts_bin', 'fc', 'ratio_bin', row='same_chr', data=df, notch=True, order=order, hue_order=hue_order, linewidth=.3,
        meanprops=MEANLINEPROPS, flierprops=FLIERPROPS, palette=get_palette_continuous(len(hue_order)), kind='box',
        height=2, aspect=2, sharey=False, legend_out=True
    )

    plt.legend(title='', prop={'size': 3}, frameon=False)

    g.map(plt.axhline, y=-1, lw=.1, ls='--', c='black', zorder=0)
    g.map(plt.axhline, y=0, lw=.1, ls='--', c='black', zorder=0)

    # plt.gcf().set_size_inches(3, 2)
    plt.savefig(f'reports/geckov2_off_target_{x}.png', bbox_inches='tight', dpi=600)
    plt.close('all')
