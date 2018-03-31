#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from scripts.plotting.svplot import import_brass_bedpe, plot_rearrangements


if __name__ == '__main__':
    svs_dir = '{}/{}.{}.csv'

    # - Import CRISPR lib
    crispr_lib = pd.read_csv(mp.LIBRARY, index_col=0).groupby('gene').agg(
        {'start': np.min, 'end': np.max}
    )

    # - Import number of chromosome copies
    chrm = pd.read_csv(mp.CN_CHR.format('wgs'), index_col=0)

    # - Import Crispy results
    svs_types = ['tandemduplication', 'deletion', 'inversion']
    fit_type = {'svs': 'SV', 'cnv': 'Copy-number', 'all': 'Both'}

    # samples = {i.split('.')[0] for i in os.listdir(mp.CRISPY_WGS_OUTDIR) if i.endswith('.csv')}

    svs_crispy = {
        t: {
            s: pd.read_csv(svs_dir.format(mp.CRISPY_WGS_OUTDIR, s, t), index_col=0).rename(columns={'fit_by': 'chr'}) for s in mp.BRCA_SAMPLES
        } for t in fit_type
    }

    # -
    varexp = pd.DataFrame({
        t: pd.DataFrame({
            c: svs_crispy[t][c].groupby('chr')['var_exp'].first() for c in svs_crispy[t]
        }).unstack() for t in svs_crispy
    })

    kmean = pd.DataFrame({
        t: pd.DataFrame({
            c: svs_crispy[t][c].groupby('chr')['k_mean'].min() for c in svs_crispy[t]
        }).unstack() for t in svs_crispy
    })

    # -
    df_svs = pd.concat([
        pd.concat([
            svs_crispy['svs'][c].query('{} == 1'.format(s)).groupby('chr')['k_mean'].mean(), chrm[c].rename('copies')
        ], axis=1).assign(sample=c).assign(sv=s) for c in svs_crispy['svs'] for s in svs_types
    ]).dropna().reset_index().rename({'index': 'chrm'}, axis=1)

    sns.jointplot('copies', 'k_mean', data=df_svs.query("sv == 'tandemduplication'"))
    plt.show()

    # -
    order = varexp.sort_values('all', ascending=False).head(10)
    # order = varexp['cnv'].sort_values(ascending=False).head(10)

    # - Variance explained
    # Build data-frame
    plot_df = varexp.loc[order.index].reset_index().rename(columns={'level_0': 'sample'})

    plot_df = pd.melt(plot_df, id_vars=['sample', 'chr'], value_vars=list(fit_type))

    plot_df = plot_df.assign(name=['{} {}'.format(s, c) for s, c in plot_df[['sample', 'chr']].values])

    plot_df = plot_df.assign(variable=plot_df['variable'].replace(fit_type).values)

    plot_df = plot_df.assign(value=(plot_df['value'] * 100).round(1).values)

    # Palette
    hue_order = list(fit_type.values())
    pal = dict(zip(*(list(fit_type.values()), sns.light_palette(bipal_dbgd[0], len(hue_order) + 1).as_hex()[1:])))

    # Plot
    g = sns.barplot('value', 'name', 'variable', data=plot_df, palette=pal, orient='h', hue_order=hue_order)

    # plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.legend(prop={'size': 5}, loc='lower right')

    g.xaxis.grid(True, color=bipal_dbgd[0], linestyle='-', linewidth=.1, alpha=.5)

    plt.xlabel('Variance explained (%)')
    plt.ylabel('')

    plt.title('Crispy fit of CRISPR/Cas9')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/crispy/brass_varexp_barplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    plot_df = pd.concat(svs_crispy['all']).sort_values('k_mean')
    plot_df = plot_df[plot_df[svs_types].sum(1) > 0]

    # -
    # sample, chrm, t = 'HCC1395', 'chr3', 'all'
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
            ax1, ax2, ax3 = plot_rearrangements(bedpe, crispr, ngsc, chrm, xlim=(120e6, 160e6), winsize=1e4)
            # ax3.set_ylim(-5, 1)
            plt.suptitle('{}'.format(sample))
            plt.gcf().set_size_inches(3, 2)
            plt.savefig('reports/crispy/brass_svplot_{}_{}_{}.png'.format(sample, chrm, t), bbox_inches='tight', dpi=600)
            plt.close('all')
