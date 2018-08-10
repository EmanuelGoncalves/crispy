#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import scripts as mp
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from natsort import natsorted
from sklearn.metrics import auc
from crispy.benchmark_plot import plot_chromosome, plot_rearrangements


MEANLINEPROPS = dict(linestyle='--', linewidth=.3, color='red')

FLIERPROPS = dict(marker='.', markersize=.1, linestyle='none', alpha=.5, lw=0)


def plot_rearrangements_seg(sample, chrm, xlim=None, import_svs='all', genes_highlight=None):
    # Import CRISRP lib
    crispr_lib = cy.get_crispr_lib().groupby('GENES').agg({'STARTpos': np.min, 'ENDpos': np.max})

    # Import BRASS bedpe
    bedpe = mp.import_brass_bedpe('{}/{}.brass.annot.bedpe'.format(mp.WGS_BRASS_BEDPE, sample), bkdist=-1, splitreads=True)

    # Import WGS sequencing depth
    ngsc = pd.read_csv(
        'data/gdsc/wgs/brass_intermediates/{}.ngscn.abs_cn.bg'.format(sample), sep='\t', header=None,
        names=['chr', 'start', 'end', 'absolute_cn', 'norm_logratio'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
    )
    ngsc = ngsc.assign(chr=ngsc['chr'].apply(lambda x: 'chr{}'.format(x)).values)

    # Import Crispy WGS fit
    crispy_sv = pd.read_csv('{}/{}.{}.csv'.format(mp.CRISPY_WGS_OUTDIR, sample, import_svs), index_col=0).rename(columns={'fit_by': 'chr'})

    # Concat CRISPR measurements
    crispr = pd.concat([crispy_sv, crispr_lib], axis=1, sort=True).dropna()

    # Plot
    ax1, ax2, ax3 = plot_rearrangements(bedpe, crispr, ngsc, chrm, winsize=1e4, show_legend=True, xlim=xlim, mark_essential=True, highlight=genes_highlight)

    return [ax1, ax2, ax3]


def plot_chromosome_from_bed(sample, chrm, genes_highlight=None):
    bed_df = pd.read_csv(f'data/crispr_bed/{sample}.crispy.bed', sep='\t').query('chr == @chrm').set_index('sgrna_id')
    bed_df = bed_df.assign(pos=bed_df[['sgrna_start', 'sgrna_end']].mean(1))

    seg = bed_df.groupby(['chr', 'start', 'end'])['cn'].first().rename('total_cn').reset_index()

    genes_highlight = [] if genes_highlight is None else bed_df[bed_df['gene'].isin(genes_highlight)].index

    ax = plot_chromosome(
        bed_df['pos'], bed_df['fc'].rename('CRISPR-Cas9'), bed_df['gp_mean'].rename('Fitted mean'),
        seg=seg, highlight=genes_highlight, chrm_cytobands=chrm, legend=True, tick_base=2, mark_essential=True
    )

    ax.set_xlabel('Position on chromosome {} (Mb)'.format(chrm), fontsize=5)
    ax.set_ylabel('')
    ax.set_title('{}'.format(sample), fontsize=7)

    return ax


def plot_chromosome_seg(sample, chrm, dtype='snp', genes_highlight=None):
    # CRISPR lib
    lib = cy.get_crispr_lib()
    lib = lib.assign(pos=lib[['STARTpos', 'ENDpos']].mean(1).values)

    # Build-frame
    plot_df = pd.read_csv('data/crispy/gdsc/crispy_crispr_{}.csv'.format(sample), index_col=0)
    plot_df = plot_df.query("fit_by == '{}'".format(chrm))
    plot_df = plot_df.assign(pos=lib.groupby('GENES')['pos'].mean().loc[plot_df.index])

    # Copy-number segmentation
    if dtype == 'snp':
        seg = pd.read_csv('data/gdsc/copynumber/snp6_bed/{}.snp6.picnic.bed'.format(sample), sep='\t')
        seg = seg[seg['#chr'] == chrm]

    elif dtype == 'wgs':
        seg = pd.read_csv('data/gdsc/wgs/ascat_bed/{}.wgs.ascat.bed.ratio.bed'.format(sample), sep='\t')
        seg = seg[seg['#chr'] == chrm]

    else:
        seg = None

    # Plot
    genes_highlight = [] if genes_highlight is None else genes_highlight

    ax = plot_chromosome(
        plot_df['pos'], plot_df['fc'].rename('CRISPR-Cas9'), plot_df['k_mean'].rename('Fitted mean'),
        seg=seg, highlight=genes_highlight, chrm_cytobands=chrm, legend=True, tick_base=2, mark_essential=True
    )

    ax.set_xlabel('Position on chromosome {} (Mb)'.format(chrm), fontsize=5)
    ax.set_ylabel('')
    ax.set_title('{}'.format(sample), fontsize=7)

    return ax, plot_df


def get_palette_continuous(n_colors, color=None):
    color = cy.PAL_DBGD[0] if color is None else color
    pal = sns.light_palette(color, n_colors=n_colors+1).as_hex()[1:]
    return pal


def bias_arocs(plot_df, x_rank='fc', groupby='cnv', min_events=10, to_plot=True):
    aucs = {}

    ax = plt.gca() if to_plot else None

    thresholds = natsorted(set(plot_df[groupby]))
    thresholds_color = get_palette_continuous(len(thresholds))

    for i, t in enumerate(thresholds):
        index_set = set(plot_df[plot_df[groupby] == t].index)

        if len(index_set) >= min_events:
            # Build data-frame
            x = plot_df[x_rank].sort_values()

            # Observed cumsum
            y = x.index.isin(index_set)
            y = np.cumsum(y) / sum(y)

            # Rank fold-changes
            x = st.rankdata(x) / x.shape[0]

            # Calculate AUC
            f_auc = auc(x, y)
            aucs[t] = f_auc

            # Plot
            if to_plot:
                ax.plot(x, y, label='{}: AURC={:.2f}'.format(t, f_auc), lw=1., c=thresholds_color[i])

    if to_plot:
        # Random
        ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

        # Limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Labels
        ax.set_xlabel('Ranked genes (fold-change)')
        ax.set_ylabel('Recall')

    return ax, aucs


def add_essential_genes_line(ax, x=0, pos=-1):
    ax.axhline(pos, ls='--', lw=.3, c=cy.PAL_DBGD[1], zorder=0)
    ax.text(x, pos, 'Essential genes', ha='left', va='bottom', fontsize=5, zorder=2, color=cy.PAL_DBGD[1])

    return ax


def bias_boxplot(plot_df, x='cnv', y='fc', despine=False, notch=True, essential=True, add_n=False, ax=None):
    ax = plt.gca() if ax is None else ax

    order = natsorted(set(plot_df[x]))
    pal = get_palette_continuous(len(order))

    sns.boxplot(
        x, y, data=plot_df, notch=notch, order=order, palette=pal, linewidth=.3,
        showmeans=True, meanline=True, meanprops=MEANLINEPROPS, flierprops=FLIERPROPS, ax=ax
    )

    if despine:
        sns.despine(ax=ax)

    if essential:
        add_essential_genes_line(ax)

    if add_n:
        for i, c in enumerate(order):
            ax.text(i, -5, 'N={}'.format((plot_df['cnv'] == c).sum()), ha='center', va='bottom', fontsize=3.5)

    return ax


def aucs_boxplot(plot_df, x='cnv', y='auc', hue=None, notch=False, essential_line=None, ax=None):
    ax = plt.gca() if ax is None else ax

    order = natsorted(set(plot_df[x]))
    order_hue = natsorted(set(plot_df[hue])) if hue is not None else None

    pal = get_palette_continuous(len(order) if hue is None else len(order_hue))

    sns.boxplot(
        x, y, hue, data=plot_df, notch=notch, order=order, hue_order=order_hue, palette=pal, linewidth=.3,
        showmeans=True, meanline=True, meanprops=MEANLINEPROPS, sym='', ax=ax
    )

    sns.stripplot(
        x, y, hue, data=plot_df, order=order, hue_order=order_hue, edgecolor='white', palette=pal,
        size=1.5, jitter=.2, dodge=True, ax=ax, linewidth=.1, alpha=.5
    )

    ax.axhline(0.5, ls='-', lw=.1, c=cy.PAL_DBGD[0], zorder=0)

    if essential_line is not None:
        add_essential_genes_line(ax, pos=essential_line)

    ax.set_ylim(0, 1)

    if hue is not None:
        handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(pal, order_hue))]
        plt.legend(handles=handles, title=hue.capitalize(), prop={'size': 6}, frameon=False).get_title().set_fontsize('6')

    ax.set_xlabel('Copy-number')
    ax.set_ylabel('Copy-number AURC (per cell line)')
    ax.set_title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    return ax
