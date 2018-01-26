#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

from sklearn.metrics import roc_auc_score


def bin_cnv(value, thresold):
    value = int(round(value, 0))
    value_bin = '{}'.format(value) if value < thresold else '{}+'.format(thresold)
    return value_bin


def multilabel_roc_auc_score(y_true, y_pred, data, min_events=1, invert=-1):
    return {
        t: roc_auc_score((data[y_true] == t).astype(int), invert * data[y_pred])
        for t in set(data[y_true]) if ((data[y_true] == t).sum() >= min_events) and (len(set(data[y_true])) > 1)
    }
