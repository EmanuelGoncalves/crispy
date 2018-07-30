#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score


def bin_bkdist(distance):
    if 1 < distance < 10e3:
        bin_distance = '1-10 kb'

    elif 10e3 < distance < 100e3:
        bin_distance = '1-100 kb'

    elif 100e3 < distance < 1e6:
        bin_distance = '0.1-1 Mb'

    elif 1e6 < distance < 10e6:
        bin_distance = '1-10 Mb'

    else:
        bin_distance = '>10 Mb'

    return bin_distance


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


def bin_cnv(value, thresold):
    if np.isfinite(value):
        value = int(round(value, 0))
        return '{}'.format(value) if value < thresold else '{}+'.format(thresold)

    else:
        return np.nan


def multilabel_roc_auc_score(y_true, y_pred, data, min_events=1, invert=-1):
    return {
        t: roc_auc_score((data[y_true] == t).astype(int), invert * data[y_pred])
        for t in set(data[y_true]) if ((data[y_true] == t).sum() >= min_events) and (len(set(data[y_true])) > 1)
    }


def multilabel_roc_auc_score_array(y_true, y_pred, min_events=1, invert=-1):
    return {
        t: roc_auc_score((y_true == t).astype(int), invert * y_pred)
        for t in set(y_true) if ((y_true == t).sum() >= min_events) and (len(set(y_true)) > 1)
    }


def qnorm(x):
    y = st.rankdata(x)
    y = -st.norm.isf(y / (len(x) + 1))
    return y


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
