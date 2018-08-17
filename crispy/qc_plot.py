#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import auc

PAL_DBGD = {0: '#656565', 1: '#F2C500', 2: '#E1E1E1'}

# - BOXPLOT PROPOS
FLIERPROPS = dict(marker='o', markerfacecolor='black', markersize=2., linestyle='none', markeredgecolor='none', alpha=1.)
MEDIANPROPS = dict(linestyle='-', linewidth=.3, color='red')
BOXPROPS = dict(linestyle='-', linewidth=.3, color='black', facecolor='white')
WHISKERPROPS = dict(linestyle='-', linewidth=.3, color='black')

# - PLOTING PROPS
SNS_RC = {
    'axes.linewidth': .3,
    'xtick.major.width': .3, 'ytick.major.width': .3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
}


class QCplot(object):
    @staticmethod
    def plot_cumsum_auc(df, index_set, ax=None, palette=None, legend=True, plot_mean=False):
        """
        Plot cumulative sum of values X considering index_set list.

        :param DataFrame df: measurements values
        :param Series index_set: list of indexes in X
        :param String ax: matplotlib Axes
        :param dict palette: palette
        :param Bool legend: Plot legend
        :param Bool plot_mean: Plot curve with the mean across all columns in X

        :return matplotlib Axes, plot_stats dict:
        """

        if ax is None:
            ax = plt.gca()

        if type(index_set) == set:
            index_set = pd.Series(list(index_set)).rename('geneset')

        plot_stats = {
            'auc': {}
        }

        # Build curve
        for f in list(df):
            # Build data-frame
            x = df[f].sort_values().dropna()

            # Observed cumsum
            y = x.index.isin(index_set)
            y = np.cumsum(y) / sum(y)

            # Rank fold-changes
            x = st.rankdata(x) / x.shape[0]

            # Calculate AUC
            f_auc = auc(x, y)
            plot_stats['auc'][f] = f_auc

            # Plot
            c = 'black' if palette is None else palette[f]
            ax.plot(x, y, label='%s: %.2f' % (f, f_auc) if (legend is True) else None, lw=1., c=c, alpha=.8)

        # Mean
        if plot_mean is True:
            x = df.mean(1).sort_values().dropna()

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
        ax.set_ylabel(f'Recall {index_set.name}')

        if legend is True:
            ax.legend(loc=4, frameon=False)

        return ax, plot_stats

    @staticmethod
    def aucs_scatter(x, y, data, rugplot=False):
        ax = plt.gca()

        ax.scatter(data[x], data[y], c='black', s=4, marker='x', lw=0.5)

        if rugplot:
            sns.rugplot(data[x], height=.02, axis='x', c='black', lw=.3, ax=ax)
            sns.rugplot(data[y], height=.02, axis='y', c='black', lw=.3, ax=ax)

        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        ax.plot(lims, lims, 'k-', lw=.3, zorder=0)

        ax.set_xlabel('{} fold-changes AURCs'.format(x.capitalize()))
        ax.set_ylabel('{} fold-changes AURCs'.format(y.capitalize()))

        return ax
