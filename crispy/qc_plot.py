#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from natsort import natsorted
from crispy.utils import Utils
from sklearn.metrics.ranking import auc

# - MAIN PALETTE
PAL_DBGD = {0: '#656565', 1: '#F2C500', 2: '#E1E1E1'}

# - BOXPLOT PROPOS
FLIERPROPS = dict(
    marker='o', markerfacecolor='black', markersize=2., linestyle='none', markeredgecolor='none', alpha=.6
)
MEDIANPROPS = dict(linestyle='-', linewidth=1., color='red')
BOXPROPS = dict(linewidth=1.)
WHISKERPROPS = dict(linewidth=1.)

# - PLOTING PROPS
SNS_RC = {
    'axes.linewidth': .3,
    'xtick.major.width': .3, 'ytick.major.width': .3,
    'xtick.major.size': 2., 'ytick.major.size': 2.,
    'xtick.direction': 'out', 'ytick.direction': 'out'
}


class QCplot(object):
    @staticmethod
    def recall_curve(rank, index_set, min_events=None):
        """
        Calculate x and y of recall curve.

        :param rank: pandas.Series

        :param index_set: pandas.Series
            indices in rank

        :param min_events: int or None, optional
            Number of minimum number of index_set to calculate curve

        :return:
        """
        x = rank.sort_values().dropna()

        # Observed cumsum
        y = x.index.isin(index_set)

        if (min_events is not None) and (sum(y) < min_events):
            return None

        y = np.cumsum(y) / sum(y)

        # Rank fold-changes
        x = st.rankdata(x) / x.shape[0]

        # Calculate AUC
        xy_auc = auc(x, y)

        return x, y, xy_auc

    @classmethod
    def recall_curve_discretise(cls, rank, discrete, thres, min_events=10):
        discrete = discrete.apply(Utils.bin_cnv, args=(thres,))

        aucs = pd.DataFrame([
            {
                discrete.name: g,
                'auc': cls.recall_curve(rank, df.index)[2]
            } for g, df in discrete.groupby(discrete) if df.shape[0] >= min_events
        ])

        return aucs

    @classmethod
    def bias_boxplot(cls, data, x='copy_number', y='auc', despine=False, notch=True, add_n=False, n_text_y=None, n_text_offset=.05, palette=None, ax=None, tick_base=.1):
        if ax is None:
            ax = plt.gca()

        order = natsorted(set(data[x]))

        if palette is None:
            palette = dict(zip(*(order, cls.get_palette_continuous(len(order)))))

        sns.boxplot(
            x, y, data=data, notch=notch, order=order, palette=palette, saturation=1., showcaps=False,
            medianprops=MEDIANPROPS, flierprops=FLIERPROPS, whiskerprops=WHISKERPROPS, boxprops=BOXPROPS, ax=ax
        )

        if despine:
            sns.despine(ax=ax)

        if add_n:
            text_y = min(data[y]) - n_text_offset if n_text_y is None else n_text_y

            for i, c in enumerate(order):
                n = (data[x] == c).sum()
                ax.text(i, text_y, f'N={n}', ha='center', va='bottom', fontsize=3.5)

            y_lim = ax.get_ylim()
            ax.set_ylim(text_y, y_lim[1])

        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base))

        return ax

    @classmethod
    def plot_cumsum_auc(cls, df, index_set, ax=None, palette=None, legend=True, plot_mean=False):
        """
        Plot cumulative sum of values X considering index_set list.

        :param DataFrame df: measurements values
        :param Series index_set: list of indexes in X
        :param ax: matplotlib Axes
        :param dict palette: palette
        :param Bool legend: Plot legend
        :param Bool plot_mean: Plot curve with the mean across all columns in X

        :return matplotlib Axes, plot_stats dict:
        """

        if ax is None:
            ax = plt.gca()

        if type(index_set) == set:
            index_set = pd.Series(list(index_set)).rename('geneset')

        plot_stats = dict(auc={})

        # Calculate recall curves
        xy_curves = {f: cls.recall_curve(df[f], index_set) for f in list(df)}

        # Build curve
        for f in xy_curves:
            x, y, xy_auc = xy_curves[f]

            # Calculate AUC
            plot_stats['auc'][f] = xy_auc

            # Plot
            c = PAL_DBGD[0] if palette is None else palette[f]
            ax.plot(x, y, label='%s: %.2f' % (f, xy_auc) if (legend is True) else None, lw=1., c=c, alpha=.8)

        # Mean
        if plot_mean is True:
            x, y, xy_auc = cls.recall_curve(df.mean(1), index_set)

            plot_stats['auc']['mean'] = xy_auc

            ax.plot(x, y, label=f'Mean: {xy_auc:.2f}', ls='--', c='red', lw=2.5, alpha=.8)

        # Random
        ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

        # Limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Labels
        ax.set_xlabel('Ranked genes (fold-change)')
        ax.set_ylabel(f'Recall {index_set.name}')

        if legend is True:
            ax.legend(loc=4, frameon=False)

        return ax, plot_stats

    @staticmethod
    def aucs_scatter(x, y, data, rugplot=False):
        ax = plt.gca()

        ax.scatter(data[x], data[y], c=PAL_DBGD[0], s=4, marker='x', lw=0.5)

        if rugplot:
            sns.rugplot(data[x], height=.02, axis='x', c=PAL_DBGD[0], lw=.3, ax=ax)
            sns.rugplot(data[y], height=.02, axis='y', c=PAL_DBGD[0], lw=.3, ax=ax)

        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, 'k-', lw=.3, zorder=0)

        ax.set_xlabel('{} fold-changes AURCs'.format(x.capitalize()))
        ax.set_ylabel('{} fold-changes AURCs'.format(y.capitalize()))

        return ax

    @staticmethod
    def get_palette_continuous(n_colors, color=PAL_DBGD[0]):
        pal = sns.light_palette(color, n_colors=n_colors + 1).as_hex()[1:]
        return pal
