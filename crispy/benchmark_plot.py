#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from crispy.utils import svtype
from matplotlib.patches import Arc
from sklearn.metrics.ranking import auc


SV_PALETTE = {
    'tandem-duplication': '#377eb8',
    'deletion': '#e41a1c',
    'translocation': '#984ea3',
    'inversion': '#4daf4a',
    'inversion_h_h': '#4daf4a',
    'inversion_t_t': '#ff7f00',
}


# TODO: Add documentation
def plot_cumsum_auc(X, index_set, ax=None, palette=None, legend=True, plot_mean=False):
    """
    Plot cumulative sum of values X considering index_set list.

    :param DataFrame X: measurements values
    :param Series index_set: list of indexes in X
    :param String ax: matplotlib Axes
    :param String cmap: Matplotlib color map
    :param Bool legend: Plot legend

    :return matplotlib Axes, plot_stats dict:
    """

    if ax is None:
        ax = plt.gca()

    plot_stats = {
        'auc': {}
    }

    palette = sns.color_palette('viridis', X.shape[1]).as_hex() if palette is None else palette

    for c, f in zip(palette, list(X)):
        # Build data-frame
        x = X[f].sort_values().dropna()

        # Observed cumsum
        y = x.index.isin(index_set)
        y = np.cumsum(y) / sum(y)

        # Rank fold-changes
        x = st.rankdata(x) / x.shape[0]

        # Calculate AUC
        f_auc = auc(x, y)
        plot_stats['auc'][f] = f_auc

        # Plot
        ax.plot(x, y, label='%s: %.2f' % (f, f_auc) if (legend is True) else None, lw=1., c=c)

    # Mean
    if plot_mean is True:
        x = X.mean(1).sort_values().dropna()

        y = x.index.isin(index_set)
        y = np.cumsum(y) / sum(y)

        x = st.rankdata(x) / x.shape[0]

        f_auc = auc(x, y)
        plot_stats['auc']['mean'] = f_auc

        ax.plot(x, y, label='Mean: %.2f' % f_auc, ls='--', c='#de2d26', lw=2.5, alpha=.8)

    # Random
    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    # Limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Labels
    ax.set_xlabel('Ranked genes (fold-change)')
    ax.set_ylabel('Recall %s' % index_set.name)

    if legend is True:
        ax.legend(loc=4)

    return ax, plot_stats


def plot_cnv_rank(x, y, ax=None, stripplot=False, hline=0.5, order=None, color='#58595B', notch=False):
    """
    Plot copy-number versus ranked y values.

    :param String x:
    :param String y:
    :param Matplotlib Axes ax:
    :param Boolean stripplot:

    :return Matplotlib Axes ax, plot_stats dict:
    """

    if ax is None:
        ax = plt.gca()

    boxplot_args = {'linewidth': .3, 'notch': notch, 'fliersize': 1, 'orient': 'v', 'sym': '' if stripplot else '.'}
    stripplot_args = {'orient': 'v', 'size': 1.5, 'linewidth': .05, 'edgecolor': 'white', 'jitter': .15, 'alpha': .75}

    y_ranked = pd.Series(st.rankdata(y) / y.shape[0], index=y.index).loc[x.index]

    sns.boxplot(x, y_ranked, ax=ax, order=order if order is not None else None, color=color, **boxplot_args)

    if stripplot:
        sns.stripplot(x, y_ranked, ax=ax, order=order if order is not None else None, color=color, **stripplot_args)

    ax.axhline(hline, lw=.1, c='black', alpha=.5)

    ax.set_xlabel('Copy-number (absolute)')
    ax.set_ylabel('Ranked %s' % y.name)

    return ax


def plot_chromosome(
        pos, original, mean, se=None, seg=None, highlight=None, ax=None, legend=False, chrm_cytobands=None,
        seg_label='Copy-number', scale=1e6, tick_base=1, legend_size=5, mark_essential=False
):

    if ax is None:
        ax = plt.gca()

    # Plot original values
    ax.scatter(pos / scale, original, s=6, marker='.', lw=0, c=cy.PAL_DBGD[1], alpha=.4, label=original.name)

    if mark_essential:
        ess = cy.get_adam_core_essential()
        ax.scatter(pos.reindex(ess) / scale, original.reindex(ess), s=6, marker='x', lw=.3, c=cy.PAL_DBGD[1], alpha=.4, edgecolors='#fc8d62', label='Core-essential')

    # Plot corrected values
    ax.scatter(pos / scale, mean, s=10, marker='.', lw=0, c='#999999', alpha=1., label=mean.name, edgecolor='#d9d9d9')

    if se is not None:
        ax.fill_between(pos / scale, mean - se, mean + se, c=cy.PAL_DBGD[1], alpha=0.2)

    # Plot segments
    if seg is not None:
        for i, (s, e, c) in enumerate(seg[['start', 'end', 'total_cn']].values):
            ax.plot([s / scale, e / scale], [c, c], lw=1.5, c=cy.PAL_DBGD[0], alpha=1., label=seg_label if i == 0 else None)

    # Highlight
    if highlight is not None:
        for ic, i in zip(*(sns.color_palette('tab20', n_colors=len(highlight)), highlight)):
            if i in pos.index:
                ax.scatter(pos.loc[i] / scale, original.loc[i], s=14, marker='X', lw=0, c=ic, alpha=.9, label=i)
    # Misc
    ax.axhline(0, lw=.3, ls='-', color='black')

    # Cytobads
    if chrm_cytobands is not None:
        cytobands = cy.get_cytobands(chrm=chrm_cytobands)

        for i, (s, e, t) in enumerate(cytobands[['start', 'end', 'band']].values):
            if t == 'acen':
                ax.axvline(s / scale, lw=.2, ls='-', color=cy.PAL_DBGD[0], alpha=.1)
                ax.axvline(e / scale, lw=.2, ls='-', color=cy.PAL_DBGD[0], alpha=.1)

            elif not i % 2:
                ax.axvspan(s / scale, e / scale, alpha=0.1, facecolor=cy.PAL_DBGD[0])

    # Legend
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': legend_size}, frameon=False)

    ax.set_xlim(pos.min() / scale, pos.max() / scale)

    ax.tick_params(axis='both', which='major', labelsize=5)

    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base))

    return ax


def plot_rearrangements(
        brass, crispr, ngsc, chrm, winsize=1e5, chrm_size=None, xlim=None, scale=1e6, show_legend=True,
        unfold_inversions=False, sv_alpha=1., sv_lw=.5, highlight=None, mark_essential=False
):
    chrm_size = cy.CHR_SIZES_HG19 if chrm_size is None else chrm_size

    #
    xlim = (0, chrm_size[chrm]) if xlim is None else xlim

    #
    intervals = np.arange(1, chrm_size[chrm] + winsize, winsize)

    # BRASS
    brass_ = brass[(brass['chr1'] == chrm) | (brass['chr2'] == chrm)]

    # NGSC
    ngsc_ = ngsc[ngsc['chr'] == chrm]
    ngsc_ = ngsc_[ngsc_['norm_logratio'] <= 2]
    ngsc_ = ngsc_.assign(location=ngsc_[['start', 'end']].mean(1))
    ngsc_ = ngsc_.assign(interval=pd.cut(ngsc_['location'], intervals, include_lowest=True, right=False))
    ngsc_ = ngsc_.groupby('interval').mean()

    # CRISPR
    crispr_ = crispr[crispr['chr'] == chrm]
    crispr_ = crispr_.assign(location=crispr_[['start', 'end']].mean(1))

    #
    if brass_.shape[0] == 0:
        return None

    # -
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [1, 2, 2]})

    #
    ax1.axhline(0.0, lw=.3, color=cy.PAL_DBGD[0])
    ax1.set_ylim(-1, 1)

    #
    ax2.scatter(ngsc_['location'] / scale, ngsc_['absolute_cn'], s=2, alpha=.8, c=cy.PAL_DBGD[0], label='Copy-number', zorder=1)
    ax2.set_ylim(0, np.ceil(ngsc_['absolute_cn'].quantile(0.9999)))

    #
    ax3.scatter(crispr_['location'] / scale, crispr_['crispr'], s=2, alpha=.8, c=cy.PAL_DBGD[1], label='CRISPR-Cas9', zorder=1)
    ax3.axhline(0.0, lw=.3, color=cy.PAL_DBGD[0])

    if mark_essential:
        ess = cy.get_adam_core_essential()
        ax3.scatter(
            crispr_.reindex(ess)['location'] / scale, crispr_.reindex(ess)['crispr'], s=12, marker='x', lw=.3, c=cy.PAL_DBGD[1],
            alpha=.4, edgecolors='#fc8d62', label='Core-essential'
        )

    # Highlight
    if highlight is not None:
        for ic, i in zip(*(sns.color_palette('tab20', n_colors=len(highlight)), highlight)):
            if i in crispr_.index:
                ax3.scatter(crispr_.loc[i, 'location'] / scale, crispr_.loc[i]['crispr'], s=14, marker='X', lw=0, c=ic, alpha=.9, label=i)

    #
    if 'cnv' in crispr.columns:
        ax2.scatter(crispr_['location'] / scale, crispr_['cnv'], s=5, alpha=1., c='#999999', zorder=2, label='ASCAT', lw=0)

    if 'k_mean' in crispr.columns:
        ax3.scatter(crispr_['location'] / scale, crispr_['k_mean'], s=5, alpha=1., c='#999999', zorder=2, label='Fitted mean', lw=0)

    #
    for c1, s1, e1, c2, s2, e2, st1, st2, sv in brass_[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'strand1', 'strand2', 'svclass']].values:
        stype = svtype(st1, st2, sv, unfold_inversions)
        stype_col = SV_PALETTE[stype]

        zorder = 3 if stype == 'tandem-duplication' else 1

        x1_mean, x2_mean = np.mean([s1, e1]), np.mean([s2, e2])

        # Plot arc
        if c1 == c2:
            angle = 0 if stype in ['tandem-duplication', 'deletion'] else 180

            xy = (np.mean([x1_mean, x2_mean]) / scale, 0)

            ax1.add_patch(
                Arc(xy, (x2_mean - x1_mean) / scale, 1., angle=angle, theta1=0, theta2=180, edgecolor=stype_col, lw=sv_lw, zorder=zorder, alpha=sv_alpha)
            )

        # Plot segments
        for ymin, ymax, ax in [(-1, 0.5, ax1), (-1, 1, ax2), (0, 1, ax3)]:
            if (c1 == chrm) and (xlim[0] <= x1_mean <= xlim[1]):
                ax.axvline(
                    x=x1_mean / scale, ymin=ymin, ymax=ymax, c=stype_col, linewidth=sv_lw, zorder=zorder, clip_on=False, label=stype, alpha=sv_alpha
                )

            if (c2 == chrm) and (xlim[0] <= x2_mean <= xlim[1]):
                ax.axvline(
                    x=x2_mean / scale, ymin=ymin, ymax=ymax, c=stype_col, linewidth=sv_lw, zorder=zorder, clip_on=False, label=stype, alpha=sv_alpha
                )

        # Translocation label
        if stype == 'translocation':
            if (c1 == chrm) and (xlim[0] <= x1_mean <= xlim[1]):
                ax1.text(x1_mean / scale, 0, ' to {}'.format(c2), color=stype_col, ha='center', fontsize=5, rotation=90, va='bottom')

            if (c2 == chrm) and (xlim[0] <= x2_mean <= xlim[1]):
                ax1.text(x2_mean / scale, 0, ' to {}'.format(c1), color=stype_col, ha='center', fontsize=5, rotation=90, va='bottom')

    #
    if show_legend:
        by_label = {l.capitalize(): p for p, l in zip(*(ax2.get_legend_handles_labels())) if l in SV_PALETTE}
        ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6}, frameon=False)

        by_label = {l: p for p, l in zip(*(ax2.get_legend_handles_labels())) if l not in SV_PALETTE}
        ax2.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6}, frameon=False)

        by_label = {l: p for p, l in zip(*(ax3.get_legend_handles_labels())) if l not in SV_PALETTE}
        ax3.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6}, frameon=False)

    #
    ax1.axis('off')

    #
    ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=2.))
    ax3.yaxis.set_major_locator(plticker.MultipleLocator(base=2.))

    #
    ax2.tick_params(axis='both', which='major', labelsize=6)
    ax3.tick_params(axis='both', which='major', labelsize=6)

    #
    ax1.set_ylabel('SV')
    ax2.set_ylabel('Copy-number', fontsize=7)
    ax3.set_ylabel('Loss of fitness', fontsize=7)

    #
    plt.xlabel('Position on chromosome {} (Mb)'.format(chrm.replace('chr', '')))

    #
    plt.xlim(xlim[0] / scale, xlim[1] / scale)

    return ax1, ax2, ax3