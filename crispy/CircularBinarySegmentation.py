#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import dpath
from CrispyPlot import MidpointNormalize, CrispyPlot


class CBS:
    @classmethod
    def run(cls, data, shuffles=int(1e3), p=0.01):
        segments = cls.segment(data, shuffles=shuffles, p=p)
        segments = cls.validate(data, segments, shuffles=shuffles, p=p)
        return segments

    @staticmethod
    def cbs_stat(x):
        """
        Given x, Compute the subinterval x[i0:i1] with the maximal segmentation statistic t.
        Returns t, i0, i1

        :param x:
        :return:
        """

        x0 = x - np.mean(x)
        n = len(x0)
        y = np.cumsum(x0)
        e0, e1 = np.argmin(y), np.argmax(y)
        i0, i1 = min(e0, e1), max(e0, e1)
        s0, s1 = y[i0], y[i1]
        return (s1 - s0) ** 2 * n / (i1 - i0 + 1) / (n + 1 - i1 + i0), i0, i1 + 1

    @staticmethod
    def tstat(x, i):
        """
        Return the segmentation statistic t testing if i is a (one-sided)  breakpoint in x

        :param x:
        :param i:
        :return:
        """
        n = len(x)
        s0 = np.mean(x[:i])
        s1 = np.mean(x[i:])
        return (n - i) * i / n * (s0 - s1) ** 2

    @classmethod
    def cbs(cls, x, shuffles=1000, p=.05):
        """
        Given x, find the interval x[i0:i1] with maximal segmentation statistic t. Test that statistic against
        given (shuffles) number of random permutations with significance p.  Return True/False, t, i0, i1; True if
        interval is significant, false otherwise.

        :param x:
        :param shuffles:
        :param p:
        :return:
        """

        max_t, max_start, max_end = cls.cbs_stat(x)
        if max_end - max_start == len(x):
            return False, max_t, max_start, max_end
        if max_start < 5:
            max_start = 0
        if len(x) - max_end < 5:
            max_end = len(x)
        thresh_count = 0
        alpha = shuffles * p
        xt = x.copy()
        for i in range(shuffles):
            np.random.shuffle(xt)
            threshold, s0, e0 = cls.cbs_stat(xt)
            if threshold >= max_t:
                thresh_count += 1
            if thresh_count > alpha:
                return False, max_t, max_start, max_end
        return True, max_t, max_start, max_end

    @classmethod
    def rsegment(cls, x, start, end, segments=None, shuffles=1000, p=.05):
        """
        Recursively segment the interval x[start:end] returning a list L of pairs (i,j) where each (i,j) is a
        significant segment.

        :param x:
        :param start:
        :param end:
        :param segments:
        :param shuffles:
        :param p:
        :return:
        """

        if segments is None:
            segments = []

        threshold, t, s, e = cls.cbs(x[start:end], shuffles=shuffles, p=p)

        if (not threshold) | (e - s < 5) | (e - s == end - start):
            segments.append((start, end))

        else:
            if s > 0:
                cls.rsegment(x, start, start + s, segments)

            if e - s > 0:
                cls.rsegment(x, start + s, start + e, segments)

            if start + e < end:
                cls.rsegment(x, start + e, end, segments)

        return segments

    @classmethod
    def segment(cls, data, shuffles, p):
        """
        Segment the array x, using significance test based on shuffles rearrangements and significance level p
        :param x:
        :param shuffles:
        :param p:
        :return:
        """

        if type(data) == pd.Series:
            data = data.values

        start, end = 0, len(data)

        segments = []

        return cls.rsegment(data, start, end, segments, shuffles=shuffles, p=p)

    @classmethod
    def validate(cls, x, L, shuffles, p):
        if type(x) == pd.Series:
            x = x.values

        S = [x[0] for x in L] + [len(x)]
        SV = [0]
        left = 0
        for test, s in enumerate(S[1:-1]):
            t = cls.tstat(x[S[left]:S[test + 2]], S[test + 1] - S[left])

            threshold = 0
            thresh_count = 0
            site = S[test + 1] - S[left]
            xt = x[S[left]:S[test + 2]].copy()
            flag = True

            for k in range(shuffles):
                np.random.shuffle(xt)
                threshold = cls.tstat(xt, site)

                if threshold > t:
                    thresh_count += 1

                if thresh_count >= p * shuffles:
                    flag = False
                    break

            if flag:
                SV.append(S[test + 1])
                left += 1

        SV.append(S[-1])

        return SV

    @staticmethod
    def generate_normal_time_series(num, minl=50, maxl=1000):
        """
        Generate a time series with num segments of minimal length minl and maximal length maxl.  Within a segment,
        data is normal with randomly chosen, normally distributed mean between -10 and 10, variance between 0 and 1.

        :param num:
        :param minl:
        :param maxl:
        :return:
        """

        data = np.array([], dtype=np.float64)
        partition = np.random.randint(minl, maxl, num)
        for p in partition:
            mean = np.random.randn() * 10
            var = np.random.randn() * 1
            if var < 0:
                var = var * -1
            tdata = np.random.normal(mean, var, p)
            data = np.concatenate((data, tdata))
        return data

    @staticmethod
    def draw_segmented_data(data, segments, ax=None):
        """
        Draw a scatterplot of the data with vertical lines at segment boundaries and horizontal lines at means of
        the segments. S is a list of segment boundaries.

        :param data:
        :param S:
        :param ax:
        :return:
        """

        if ax is None:
            ax = plt.gca()

        ax.scatter(range(len(data)), data.values, s=1, c=data, cmap='RdYlBu', norm=MidpointNormalize(midpoint=0.))

        ax.axhline(0, ls='-', lw=.3, c=CrispyPlot.PAL_DBGD[2], zorder=0)

        for x in segments:
            ax.axvline(x, ls='--', lw=.3, c=CrispyPlot.PAL_DBGD[2], zorder=0)

        for i in range(1, len(segments)):
            y = np.mean(data.values[segments[i - 1]:segments[i]])
            ax.hlines(y, segments[i - 1], segments[i], color=CrispyPlot.PAL_DBGD[1], lw=1.)

        ax.set_title(data.name)

        return ax


if __name__ == '__main__':
    fc = pd.read_csv(f"{dpath}/example_ccleanr.csv")

    segs = CBS.run(fc.query("CHR == '1'")['avgFC'], shuffles=int(1e5), p=0.01)

