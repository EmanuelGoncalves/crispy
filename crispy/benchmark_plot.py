#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from crispy import PAL_DBGD
from sklearn.metrics.ranking import auc

CYTOBANDS_FILE = 'data/cytoBand.txt'


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
    ax.set_xlabel('Ranked')
    ax.set_ylabel('Recall %s' % index_set.name)

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


def plot_chromosome(pos, original, mean, se=None, seg=None, highlight=None, ax=None, legend=False, cytobands=True, seg_label='Copy-number', scale=1e6, tick_base=1, legend_size=5):
    if ax is None:
        ax = plt.gca()

    # Plot original values
    ax.scatter(pos / scale, original, s=6, marker='.', lw=0, c=PAL_DBGD[1], alpha=.8, label=original.name)

    # Plot corrected values
    ax.scatter(pos / scale, mean, s=10, marker='.', lw=0, c='#999999', alpha=1., label=mean.name, edgecolor='#d9d9d9')

    if se is not None:
        ax.fill_between(pos / scale, mean - se, mean + se, c=PAL_DBGD[1], alpha=0.2)

    # Plot segments
    if seg is not None:
        for i, (s, e, c) in enumerate(seg[['start', 'end', 'total_cn']].values):
            ax.plot([s / scale, e / scale], [c, c], lw=1.5, c=PAL_DBGD[0], alpha=1., label=seg_label if i == 0 else None)

    # Highlight
    if highlight is not None:
        for ic, i in zip(*(sns.color_palette('tab20', n_colors=len(highlight)), highlight)):
            if i in pos.index:
                ax.scatter(pos.loc[i] / scale, original.loc[i], s=14, marker='X', lw=0, c=ic, alpha=.9, label=i)
    # Misc
    ax.axhline(0, lw=.3, ls='-', color='black')

    # Cytobads
    if cytobands is not None:
        for i, (s, e, t) in enumerate(cytobands[['start', 'end', 'band']].values):
            if t == 'acen':
                ax.axvline(s / scale, lw=.2, ls='-', color=PAL_DBGD[0], alpha=.1)
                ax.axvline(e / scale, lw=.2, ls='-', color=PAL_DBGD[0], alpha=.1)

            elif not i % 2:
                ax.axvspan(s / scale, e / scale, alpha=0.1, facecolor=PAL_DBGD[0])

    # Legend
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': legend_size})

    ax.set_xlim(pos.min() / scale, pos.max() / scale)

    ax.tick_params(axis='both', which='major', labelsize=5)

    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base))

    return ax


def import_cytobands(file=None, chrm=None):
    file = CYTOBANDS_FILE if file is None else file

    cytobands = pd.read_csv(file, sep='\t')

    if chrm is not None:
        cytobands = cytobands[cytobands['chr'] == chrm]

    assert cytobands.shape[0] > 0, '{} not found in cytobands file'

    return cytobands
