#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score


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


def qnorm(x):
    """ Quantile normalisation function imported from limix scipy-sugar:
    https://github.com/limix/scipy-sugar
    """

    x = np.asarray(x, float).copy()

    ok = np.isfinite(x)

    x[ok] *= -1

    y = np.empty_like(x)

    y[ok] = st.rankdata(x[ok])

    y[ok] = st.norm.isf(y[ok] / (sum(ok) + 1))

    y[~ok] = x[~ok]

    return y
