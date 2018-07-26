#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from natsort import natsorted
from sklearn.metrics import auc
import matplotlib.patches as mpatches


MEANLINEPROPS = dict(linestyle='--', linewidth=.3, color='red')
FLIERPROPS = dict(marker='o', markersize=.1, linestyle='none', alpha=.5, lw=0)


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


def bias_boxplot(plot_df, x='cnv', y='fc', despine=True, notch=True, essential=True, add_n=False, ax=None):
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

def aucs_boxplot(plot_df, x='cnv', y='auc', hue=None, notch=False, ax=None):
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
        size=1, jitter=.2, dodge=True, ax=ax, linewidth=.1, alpha=.5
    )

    ax.axhline(0.5, ls='-', lw=.1, c=cy.PAL_DBGD[0], zorder=0)

    ax.set_ylim(0, 1)

    if hue is not None:
        handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(pal, order_hue))]
        plt.legend(handles=handles, title=hue.capitalize(), prop={'size': 6}, frameon=False).get_title().set_fontsize('6')

    ax.set_xlabel('Copy-number')
    ax.set_ylabel('Copy-number AURC (per cell line)')
    ax.set_title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    return ax
