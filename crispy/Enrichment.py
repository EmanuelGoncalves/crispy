#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import random
import logging
import operator
import numpy as np
import pandas as pd
import pkg_resources
from crispy import CrispyPlot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats.distributions import hypergeom
from statsmodels.stats.multitest import multipletests


dpath = pkg_resources.resource_filename("crispy", "data/")


class SSGSEA(object):
    @classmethod
    def gsea(cls, dataset, signature, permutations=0):
        # Sort data-set by values
        _dataset = list(
            zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=True))
        )
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

            e_score = cls.__es(
                genes, expression, _signature, nr, nh, n, hits, running_hit
            )

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

                # If no permutation was above the Enrichment score the p-value is lower than 1 divided by the number
                # of permutations
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


class Enrichment:
    """
    Gene enrichment analysis class.

    """

    def __init__(
        self, gmts, sig_min_len=5, verbose=0, padj_method="fdr_bh", permutations=0
    ):

        self.verbose = verbose
        self.padj_method = padj_method
        self.permutations = permutations

        self.sig_min_len = sig_min_len

        self.gmts = {f: self.read_gmt(f"{dpath}/pathways/{f}") for f in gmts}

    def __assert_gmt_file(self, gmt_file):
        assert gmt_file in self.gmts, f"{gmt_file} not in gmt files: {self.gmts.keys()}"

    def __assert_signature(self, gmt_file, signature):
        self.__assert_gmt_file(gmt_file)
        assert signature in self.gmts[gmt_file], f"{signature} not in {gmt_file}"

    @staticmethod
    def read_gmt(file_path):
        with open(file_path) as f:
            signatures = {
                l.split("\t")[0]: set(l.strip().split("\t")[2:]) for l in f.readlines()
            }
        return signatures

    def gsea(self, values, signature):
        return SSGSEA.gsea(values.to_dict(), signature, permutations=self.permutations)

    def gsea_enrichments(self, values, gmt_file):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        if self.verbose > 0 and type(values) == pd.Series:
            logging.getLogger("DTrace").info(f"Values={values.name}")

        ssgsea = []
        for gset in geneset:
            if self.verbose > 1:
                logging.getLogger("DTrace").info(f"Gene-set={gset}")

            gset_len = len({i for i in geneset[gset] if i in values.index})

            e_score, p_value, _, _ = self.gsea(values, geneset[gset])

            ssgsea.append(
                dict(gset=gset, e_score=e_score, p_value=p_value, len=gset_len)
            )

        ssgsea = pd.DataFrame(ssgsea).set_index("gset").sort_values("e_score")

        if self.sig_min_len is not None:
            ssgsea = ssgsea.query(f"len >= {self.sig_min_len}")

        if self.permutations > 0:
            ssgsea["adj.p_value"] = multipletests(
                ssgsea["p_value"], method=self.padj_method
            )[1]

        return ssgsea

    def get_signature(self, gmt_file, signature):
        self.__assert_signature(gmt_file, signature)
        return self.gmts[gmt_file][signature]

    def plot(self, values, gmt_file, signature, vertical_lines=False, shade=False):
        if type(signature) == str:
            signature = self.get_signature(gmt_file, signature)

        e_score, p_value, hits, running_hit = self.gsea(values, signature)

        ax = GSEAplot.plot_gsea(
            hits,
            running_hit,
            dataset=values.to_dict(),
            vertical_lines=vertical_lines,
            shade=shade,
        )

        return ax

    @staticmethod
    def hypergeom_test(signature, background, sublist):
        """
        Performs hypergeometric test

        Arguements:
                signature: {string} - Signature IDs
                background: {string} - Background IDs
                sublist: {string} - Sub-set IDs

        # hypergeom.sf(x, M, n, N, loc=0)
        # M: total number of objects,
        # n: total number of type I objects
        # N: total number of type I objects drawn without replacement

        """
        pvalue = hypergeom.sf(
            len(sublist.intersection(signature)),
            len(background),
            len(background.intersection(signature)),
            len(sublist),
        )

        intersection = len(sublist.intersection(signature))

        return pvalue, intersection

    def hypergeom_enrichments(self, sublist, background, gmt_file):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        ssgsea_geneset = []
        for gset in geneset:
            if self.verbose > 0:
                logging.getLogger("DTrace").info(f"Gene-set={gset}")

            p_value, intersection = self.hypergeom_test(
                signature=geneset[gset], background=background, sublist=sublist
            )

            ssgsea_geneset.append(
                dict(
                    gset=gset,
                    p_value=p_value,
                    len_sig=len(geneset[gset]),
                    len_intersection=intersection,
                )
            )

        ssgsea_geneset = (
            pd.DataFrame(ssgsea_geneset).set_index("gset").sort_values("p_value")
        )
        ssgsea_geneset["adj.p_value"] = multipletests(
            ssgsea_geneset["p_value"], method=self.padj_method
        )[1]

        return ssgsea_geneset

    @staticmethod
    def one_sided_pvalue(escore, escores):
        count = np.sum((escores >= escore) if escore >= 0 else (escores <= escore))

        p_value = 1 / len(escores) if count == 0 else count / len(escores)

        return p_value


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
