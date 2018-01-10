#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

from numpy import isnan
from sklearn.linear_model import LinearRegression


def lr(xx, yy):
    xx, yy = xx[~isnan(yy)], yy[~isnan(yy)]

    lm = LinearRegression().fit(xx, yy)

    lm_r2 = lm.score(xx, yy)

    return {'r2': lm_r2}
