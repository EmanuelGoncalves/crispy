#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
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
