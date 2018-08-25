#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from crispy.qc_plot import QCplot
from matplotlib.gridspec import GridSpec


class SSGSEA(object):

    @classmethod
    def gsea(cls, dataset, signature, permutations=0):
        # Sort data-set by values
        _dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=True)))
        genes, expression = _dataset[0], _dataset[1]

        # Signature overlapping the data-set
        _signature = set(signature).intersection(genes)

        # Check signature overlap
        e_score, p_value = np.NaN, np.NaN
        hits, running_hit = [], []
        if len(_signature) != 0:

            # ---- Calculate signature enrichment score
            n, sig_size = len(genes), len(_signature)
            nh = n - sig_size
            nr = sum([abs(dataset[g]) for g in _signature])

            e_score = cls.__es(genes, expression, _signature, nr, nh, n, hits, running_hit)

            # ---- Calculate statistical enrichment
            # Generate random signatures sampled from the data-set genes
            if permutations > 0:
                count = 0

                for i in range(permutations):
                    r_signature = random.sample(genes, sig_size)

                    r_nr = sum([abs(dataset[g]) for g in r_signature])

                    r_es = cls.__es(genes, expression, r_signature, r_nr, nh, n)

                    if (r_es >= e_score >= 0) or (r_es <= e_score < 0):
                        count += 1

                # If no permutation was above the Enrichment score the p-value is lower than 1 divided by the number of permutations
                p_value = 1 / permutations if count == 0 else count / permutations

            else:
                p_value = np.nan

        return e_score, p_value, hits, running_hit

    @staticmethod
    def __es(genes, expression, signature, nr, nh, n, hits=None, running_hit=None):
        hit, miss, es, r = 0, 0, 0, 0
        for i in range(n):
            if genes[i] in signature:
                hit += abs(expression[i]) / nr

                if hits is not None:
                    hits.append(1)

            else:
                miss += 1 / nh

                if hits is not None:
                    hits.append(0)

            r = hit - miss

            if running_hit is not None:
                running_hit.append(r)

            if abs(r) > abs(es):
                es = r

        return es

    @classmethod
    def plot_gsea(cls, hits, running_hit, dataset=None, vertical_lines=False, shade=False):
        x, y = np.array(range(len(hits))), np.array(running_hit)

        if dataset is not None:
            gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.1)
            axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

        else:
            axs = [plt.gca()]

        # GSEA running hit
        axs[0].plot(x, y, '-', c=QCplot.PAL_DBGD[0])

        if shade:
            axs[0].fill_between(x, 0, y, alpha=.5, color=QCplot.PAL_DBGD[2])

        if vertical_lines:
            for i in x[np.array(hits, dtype='bool')]:
                axs[0].axvline(i, c=QCplot.PAL_DBGD[0], lw=.3, alpha=.2, zorder=0)

        axs[0].axhline(0, c=QCplot.PAL_DBGD[0], lw=.1, ls='-')
        axs[0].set_ylabel('Enrichment score')
        axs[0].get_xaxis().set_visible(False)
        axs[0].set_xlim([0, len(x)])

        if dataset is not None:
            dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=False)))

            # Data
            axs[1].scatter(x, dataset[1], c=QCplot.PAL_DBGD[0], linewidths=0, s=2)

            if shade:
                axs[1].fill_between(x, 0, dataset[1], alpha=.5, color=QCplot.PAL_DBGD[2])

            if vertical_lines:
                for i in x[np.array(hits, dtype='bool')]:
                    axs[1].axvline(i, c=QCplot.PAL_DBGD[0], lw=.3, alpha=.2, zorder=0)

            axs[1].axhline(0, c='black', lw=.3, ls='-')
            axs[1].set_ylabel('Data value')
            axs[1].get_xaxis().set_visible(False)
            axs[1].set_xlim([0, len(x)])

        return axs[0] if dataset is None else axs
