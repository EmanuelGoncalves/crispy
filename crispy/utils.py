#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import scipy.stats as st


class Utils(object):
    @staticmethod
    def bin_bkdist(distance):
        """
        Discretise genomic distances

        :param distance: float

        :return:

        """
        if distance == -1:
            bin_distance = 'diff. chr.'

        elif distance == 0:
            bin_distance = '0'

        elif 1 < distance < 10e3:
            bin_distance = '1-10 kb'

        elif 10e3 < distance < 100e3:
            bin_distance = '10-100 kb'

        elif 100e3 < distance < 1e6:
            bin_distance = '0.1-1 Mb'

        elif 1e6 < distance < 10e6:
            bin_distance = '1-10 Mb'

        else:
            bin_distance = '>10 Mb'

        return bin_distance

    @staticmethod
    def svtype(strand1, strand2, svclass, unfold_inversions):
        if svclass == 'translocation':
            svtype = 'translocation'

        elif strand1 == '+' and strand2 == '+':
            svtype = 'deletion'

        elif strand1 == '-' and strand2 == '-':
            svtype = 'tandem-duplication'

        elif strand1 == '+' and strand2 == '-':
            svtype = 'inversion_h_h' if unfold_inversions else 'inversion'

        elif strand1 == '-' and strand2 == '+':
            svtype = 'inversion_t_t' if unfold_inversions else 'inversion'

        else:
            assert False, 'SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}'.format(strand1, strand2, svclass)

        return svtype

    @staticmethod
    def bin_cnv(value, thresold):
        if not np.isfinite(value):
            return np.nan

        value = int(round(value, 0))
        value = f'{value}' if value < thresold else f'{thresold}+'

        return value

    @staticmethod
    def qnorm(x):
        y = st.rankdata(x)
        y = -st.norm.isf(y / (len(x) + 1))
        return y


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
