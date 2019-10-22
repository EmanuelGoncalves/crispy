#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.CrispyPlot import CrispyPlot


class LibraryRepresentaion:
    def __init__(self, counts, percentiles=None):
        self.counts = counts
        self.percentiles = [2.5, 97.5] if percentiles is None else percentiles

    def gini(self):
        gini_scores = {}

        for c in self.counts:
            values = self.counts[c].dropna()

            sorted_list = sorted(values)

            height, area = 0.0, 0.0

            for value in sorted_list:
                height += value
                area += height - value / 2.0

            fair_area = height * len(values) / 2.0

            gini_scores[c] = (fair_area - area) / fair_area

        return pd.Series(gini_scores)

    def percentile(self):
        percentile_ranges = {}

        for c in self.counts:
            values = self.counts[c]

            p_low, p_high = np.nanpercentile(np.log10(values), self.percentiles)

            percentile_range = p_high / p_low

            percentile_ranges[c] = percentile_range

        return pd.Series(percentile_ranges)

    def dropout_rate(self):
        dropout_rates = {}

        for c in self.counts:
            values = self.counts[c]
            dropout_rates[c] = sum(values == 0) / len(values)

        return pd.Series(dropout_rates)

    def boxplot(self, order=None, palette=None, ax=None):
        values = np.log10(self.counts)

        if ax is None:
            ax = plt.gca()

        sns.boxplot(
            data=values,
            order=order,
            orient="h",
            palette=palette,
            saturation=1,
            showcaps=False,
            boxprops=dict(linewidth=0.3),
            whiskerprops=dict(linewidth=0.3),
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markersize=1.0,
                linestyle="none",
                markeredgecolor="none",
                alpha=0.6,
            ),
            notch=True,
            ax=ax,
        )

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        ax.set_xlabel("sgRNAs counts (log10)")

        return ax

    def distplot(self, palette=None, ax=None):
        if ax is None:
            ax = plt.gca()

        for c in self.counts:
            values = self.counts[c]

            sns.distplot(
                values,
                hist=False,
                kde_kws=dict(cut=0),
                label=c,
                color=None if palette is None else palette[c],
                ax=ax,
            )

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        ax.set_xlabel("sgRNA counts")

        plt.legend(frameon=False, prop={'size': 4})

        return ax

    def lorenz_curve(self, palette=None, ax=None):
        if ax is None:
            ax = plt.gca()

        gini_scores = self.gini()

        for c in self.counts:
            values = np.sort(self.counts[c].dropna())

            x_lorenz = np.cumsum(values) / np.sum(values)
            y_lorenz = np.arange(1, len(x_lorenz) + 1) / len(x_lorenz)

            ax.plot(
                y_lorenz,
                x_lorenz,
                color=CrispyPlot.PAL_DBGD[0] if palette is None else palette[c],
                label=f"{c} (gini={gini_scores[c]:.2f})",
                zorder=2,
                alpha=.5,
            )

        ax.plot([0, 1], [0, 1], "k-", lw=0.3, zorder=1)

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        ax.set_ylabel("Fraction of total reads")
        ax.set_xlabel("Fraction of total sgRNAs")

        plt.legend(loc=2, frameon=False, prop={"size": 4})

        return ax
