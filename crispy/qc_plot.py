#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import auc


class QCplot(object):
    @staticmethod
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

        if type(index_set) == set:
            index_set = pd.Series(list(index_set)).rename('geneset')

        plot_stats = {
            'auc': {}
        }

        # Create palette
        if palette is None:
            palette = sns.color_palette('viridis', X.shape[1]).as_hex()
            palette = dict(zip(*(X.columns, palette)))

        # Build curve
        for f in list(X):
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
            ax.plot(x, y, label='%s: %.2f' % (f, f_auc) if (legend is True) else None, lw=1., c=palette[f])

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
        ax.set_ylabel(f'Recall {index_set.name}')

        if legend is True:
            ax.legend(loc=4, frameon=False)

        return ax, plot_stats
