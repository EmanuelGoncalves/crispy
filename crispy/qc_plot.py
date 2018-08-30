#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import operator
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from crispy import Utils
from natsort import natsorted
from matplotlib.patches import Arc
from collections import OrderedDict
from sklearn.metrics.ranking import auc
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score


class CrispyPlot(object):
    # PLOTING PROPS
    SNS_RC = {
        'axes.linewidth': .3,
        'xtick.major.width': .3, 'ytick.major.width': .3,
        'xtick.major.size': 2., 'ytick.major.size': 2.,
        'xtick.direction': 'out', 'ytick.direction': 'out'
    }

    # PALETTES
    PAL_DBGD = {0: '#656565', 1: '#F2C500', 2: '#E1E1E1'}

    SV_PALETTE = {
        'tandem-duplication': '#377eb8',
        'deletion': '#e41a1c',
        'translocation': '#984ea3',
        'inversion': '#4daf4a',
        'inversion_h_h': '#4daf4a',
        'inversion_t_t': '#ff7f00',
    }

    # BOXPLOT PROPOS
    FLIERPROPS = dict(
        marker='o', markerfacecolor='black', markersize=2., linestyle='none', markeredgecolor='none', alpha=.6
    )
    MEDIANPROPS = dict(linestyle='-', linewidth=1., color='red')
    BOXPROPS = dict(linewidth=1.)
    WHISKERPROPS = dict(linewidth=1.)

    @staticmethod
    def get_palette_continuous(n_colors, color=PAL_DBGD[0]):
        pal = sns.light_palette(color, n_colors=n_colors + 2).as_hex()[2:]
        return pal


class QCplot(CrispyPlot):
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

    @staticmethod
    def pr_curve(rank, true_set=None, false_set=None, min_events=10):
        if true_set is None:
            true_set = Utils.get_essential_genes(return_series=False)

        if false_set is None:
            false_set = Utils.get_non_essential_genes(return_series=False)

        index_set = true_set.union(false_set)

        rank = rank[rank.index.isin(index_set)]

        if len(rank) == 0:
            return np.nan

        y_true = rank.index.isin(true_set).astype(int)

        if sum(y_true) < min_events:
            return np.nan

        return roc_auc_score(y_true, -rank)

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
    def bias_boxplot(
            cls, data, x='copy_number', y='auc', hue=None, despine=False, notch=True, add_n=False, n_text_y=None,
            n_text_offset=.05, palette=None, ax=None, tick_base=.1, hue_order=None
    ):
        if ax is None:
            ax = plt.gca()

        order = natsorted(set(data[x]))

        if palette is None and hue is not None:
            hue_order = natsorted(set(data[hue])) if hue_order is None else hue_order
            palette = dict(zip(*(hue_order, cls.get_palette_continuous(len(hue_order)))))

        elif palette is None:
            palette = dict(zip(*(order, cls.get_palette_continuous(len(order)))))

        sns.boxplot(
            x, y, data=data, hue=hue, notch=notch, order=order, palette=palette, saturation=1., showcaps=False, hue_order=hue_order,
            medianprops=cls.MEDIANPROPS, flierprops=cls.FLIERPROPS, whiskerprops=cls.WHISKERPROPS, boxprops=cls.BOXPROPS, ax=ax
        )

        if despine:
            sns.despine(ax=ax)

        if add_n:
            text_y = min(data[y]) - n_text_offset if n_text_y is None else n_text_y

            for i, c in enumerate(order):
                n = np.sum(data[x] == c)
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
            c = cls.PAL_DBGD[0] if palette is None else palette[f]
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

    @classmethod
    def aucs_scatter(cls, x, y, data, diagonal_line=True, rugplot=False, s=4, marker='x', lw=.5):
        ax = plt.gca()

        ax.scatter(data[x], data[y], c=cls.PAL_DBGD[0], s=s, marker=marker, lw=lw)

        if rugplot:
            sns.rugplot(data[x], height=.02, axis='x', c=cls.PAL_DBGD[0], lw=.3, ax=ax)
            sns.rugplot(data[y], height=.02, axis='y', c=cls.PAL_DBGD[0], lw=.3, ax=ax)

        if diagonal_line:
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            ax.plot(lims, lims, 'k-', lw=.3, zorder=0)

        ax.set_xlabel('{} fold-changes AURCs'.format(x.capitalize()))
        ax.set_ylabel('{} fold-changes AURCs'.format(y.capitalize()))

        return ax

    @classmethod
    def aucs_scatter_pairgrid(cls, data, set_lim=True, axhlines=True, axvlines=True):
        g = sns.PairGrid(data, despine=False)

        g.map_offdiag(plt.scatter, color=cls.PAL_DBGD[0], s=4, marker='o', lw=0.5, alpha=.7)

        g.map_diag(plt.hist, color=cls.PAL_DBGD[0])

        if set_lim:
            g.set(ylim=(0, 1), xlim=(0, 1))

        for ax_indices in [np.tril_indices_from(g.axes, -1), np.triu_indices_from(g.axes, 1)]:
            for i, j in zip(*ax_indices):
                if axhlines:
                    g.axes[i, j].axhline(.5, ls=':', lw=.3, zorder=0, color='black')

                if axvlines:
                    g.axes[i, j].axvline(.5, ls=':', lw=.3, zorder=0, color='black')

        for ax_indices in [np.tril_indices_from(g.axes, -1), np.triu_indices_from(g.axes, 1)]:
            for i, j in zip(*ax_indices):
                (x0, x1), (y0, y1) = g.axes[i, j].get_xlim(), g.axes[i, j].get_ylim()
                lims = [max(x0, y0), min(x1, y1)]

                g.axes[i, j].plot(lims, lims, 'k-', lw=.3, zorder=0)

        return g

    @classmethod
    def plot_chromosome(
            cls, crispy_bed, ascat_bed, chrm, highlight=None, ax=None, legend=False, scale=1e6, tick_base=1, legend_size=5
    ):

        if ax is None:
            ax = plt.gca()

        # - Build data-frames
        # ASCAT
        ascat_ = ascat_bed.query(f"chr == '{chrm}'")

        # CRISPR
        crispr_ = crispy_bed[crispy_bed['chr'] == chrm]
        crispr_ = crispr_.assign(location=crispr_[['sgrna_start', 'sgrna_end']].mean(1))

        crispr_gene_ = crispr_.groupby('gene')[['fold_change', 'location']].mean()

        # Plot original values
        ax.scatter(crispr_['location'] / scale, crispr_['fold_change'], s=6, marker='.', lw=0, c=cls.PAL_DBGD[1], alpha=.4, label='CRISPR-Cas9')

        # Segment mean
        for (s, e), gp_mean in crispr_.groupby(['start', 'end'])['fold_change']:
            ax.plot((s / scale, e / scale), (gp_mean.mean(), gp_mean.mean()), alpha=1., c=cls.PAL_DBGD[2], zorder=3, label='Segment mean', lw=2)

        # Plot segments
        for s, e, cn in ascat_[['start', 'end', 'copy_number']].values:
            ax.plot((s / scale, e / scale), (cn, cn), alpha=1., c=cls.PAL_DBGD[0], zorder=3, label='ASCAT', lw=2)

        # Highlight
        if highlight is not None:
            for ic, i in zip(*(sns.color_palette('tab20', n_colors=len(highlight)), highlight)):
                if i in crispr_gene_.index:
                    ax.scatter(
                        crispr_gene_['location'].loc[i] / scale, crispr_gene_['fold_change'].loc[i], s=14, marker='X', lw=0, c=ic, alpha=.9, label=i
                    )
        # Misc
        ax.axhline(0, lw=.3, ls='-', color='black')

        # Cytobads
        cytobands = Utils.get_cytobands(chrm=chrm)

        for i, (s, e, t) in enumerate(cytobands[['start', 'end', 'band']].values):
            if t == 'acen':
                ax.axvline(s / scale, lw=.2, ls='-', color=cls.PAL_DBGD[0], alpha=.1)
                ax.axvline(e / scale, lw=.2, ls='-', color=cls.PAL_DBGD[0], alpha=.1)

            elif not i % 2:
                ax.axvspan(s / scale, e / scale, alpha=0.1, facecolor=cls.PAL_DBGD[0])

        # Legend
        if legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend()
            ax.legend(
                by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': legend_size}, frameon=False
            )

        ax.set_xlim(crispr_['start'].min() / scale, crispr_['end'].max() / scale)

        ax.tick_params(axis='both', which='major', labelsize=5)

        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base))

        return ax

    @classmethod
    def plot_rearrangements(
            cls, brass_bedpe, ascat_bed, crispy_bed, chrm, chrm_size=None, xlim=None, scale=1e6, show_legend=True,
            unfold_inversions=False, sv_alpha=1., sv_lw=.3, highlight=None, mark_essential=False
    ):
        # - Define default params
        chrm_size = Utils.CHR_SIZES_HG19 if chrm_size is None else chrm_size

        xlim = (0, chrm_size[chrm]) if xlim is None else xlim

        # - Build data-frames
        # BRASS
        brass_ = brass_bedpe[(brass_bedpe['chr1'] == chrm) | (brass_bedpe['chr2'] == chrm)]

        # ASCAT
        ascat_ = ascat_bed.query(f"chr == '{chrm}'")

        # CRISPR
        crispr_ = crispy_bed[crispy_bed['chr'] == chrm]
        crispr_ = crispr_.assign(location=crispr_[['sgrna_start', 'sgrna_end']].mean(1))

        crispr_gene_ = crispr_.groupby('gene')[['fold_change', 'location']].mean()

        if brass_.shape[0] == 0:
            return None, None, None

        # - Plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', gridspec_kw={'height_ratios': [1, 2, 2]})

        # Top panel
        ax1.axhline(0.0, lw=.3, color=cls.PAL_DBGD[0])
        ax1.set_ylim(-1, 1)

        # Middle panel
        for i, (_, s, e, cn) in ascat_.iterrows():
            ax2.plot((s / scale, e / scale), (cn, cn), alpha=1., c=cls.PAL_DBGD[2], zorder=3, label='ASCAT', lw=2)

        # Bottom panel
        ax3.scatter(crispr_['location'] / scale, crispr_['fold_change'], s=1, alpha=.5, lw=0, c=cls.PAL_DBGD[1], label='CRISPR-Cas9', zorder=1)
        ax3.axhline(0.0, lw=.3, color=cls.PAL_DBGD[0])

        for (s, e), gp_mean in crispr_.groupby(['start', 'end'])['fold_change']:
            ax3.plot((s / scale, e / scale), (gp_mean.mean(), gp_mean.mean()), alpha=1., c=cls.PAL_DBGD[2], zorder=3, label='Segment mean', lw=2)

        if mark_essential:
            ess = Utils.get_adam_core_essential()
            ax3.scatter(
                crispr_gene_.reindex(ess)['location'] / scale, crispr_gene_.reindex(ess)['fold_change'], s=5, marker='x', lw=.3, c=cls.PAL_DBGD[1],
                alpha=.4, edgecolors='#fc8d62', label='Core-essential'
            )

        # Highlight
        if highlight is not None:
            for ic, i in zip(*(sns.color_palette('tab20', n_colors=len(highlight)), highlight)):
                if i in crispr_.index:
                    ax3.scatter(crispr_.loc[i, 'location'] / scale, crispr_.loc[i]['fold_change'], s=14, marker='X', lw=0, c=ic, alpha=.9, label=i)

        #
        for c1, s1, e1, c2, s2, e2, st1, st2, sv in brass_[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'strand1', 'strand2', 'svclass']].values:
            stype = Utils.svtype(st1, st2, sv, unfold_inversions)
            stype_col = cls.SV_PALETTE[stype]

            zorder = 2 if stype == 'tandem-duplication' else 1

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
            by_label = {l.capitalize(): p for p, l in zip(*(ax2.get_legend_handles_labels())) if l in cls.SV_PALETTE}
            ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6}, frameon=False)

            by_label = {l: p for p, l in zip(*(ax2.get_legend_handles_labels())) if l not in cls.SV_PALETTE}
            ax2.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6}, frameon=False)

            by_label = {l: p for p, l in zip(*(ax3.get_legend_handles_labels())) if l not in cls.SV_PALETTE}
            ax3.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6}, frameon=False)

        #
        ax1.axis('off')

        #
        ax2.set_ylim(0, np.ceil(ascat_['copy_number'].quantile(0.9999) + .5))

        #
        ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=2.))
        ax3.yaxis.set_major_locator(plticker.MultipleLocator(base=1.))

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


class GSEAplot(CrispyPlot):
    @classmethod
    def plot_gsea(cls, hits, running_hit, dataset=None, vertical_lines=False, shade=False):
        x, y = np.array(range(len(hits))), np.array(running_hit)

        if dataset is not None:
            gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.1)
            axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

        else:
            axs = [plt.gca()]

        # GSEA running hit
        axs[0].plot(x, y, '-', c=cls.PAL_DBGD[0])

        if shade:
            axs[0].fill_between(x, 0, y, alpha=.5, color=cls.PAL_DBGD[2])

        if vertical_lines:
            for i in x[np.array(hits, dtype='bool')]:
                axs[0].axvline(i, c=cls.PAL_DBGD[0], lw=.3, alpha=.2, zorder=0)

        axs[0].axhline(0, c=cls.PAL_DBGD[0], lw=.1, ls='-')
        axs[0].set_ylabel('Enrichment score')
        axs[0].get_xaxis().set_visible(False)
        axs[0].set_xlim([0, len(x)])

        if dataset is not None:
            dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=False)))

            # Data
            axs[1].scatter(x, dataset[1], c=cls.PAL_DBGD[0], linewidths=0, s=2)

            if shade:
                axs[1].fill_between(x, 0, dataset[1], alpha=.5, color=cls.PAL_DBGD[2])

            if vertical_lines:
                for i in x[np.array(hits, dtype='bool')]:
                    axs[1].axvline(i, c=cls.PAL_DBGD[0], lw=.3, alpha=.2, zorder=0)

            axs[1].axhline(0, c='black', lw=.3, ls='-')
            axs[1].set_ylabel('Data value')
            axs[1].get_xaxis().set_visible(False)
            axs[1].set_xlim([0, len(x)])

        return axs[0] if dataset is None else axs
