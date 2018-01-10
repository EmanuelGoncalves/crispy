# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
cimport numpy as np
from sklearn.linear_model import LinearRegression


def lr(xs, ys):
    cdef int x_idx, y_idx

    cdef np.ndarray r2_scores = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    cdef np.ndarray xx = np.zeros(ys.shape[0], dtype=np.float64)
    cdef np.ndarray yy = np.zeros(ys.shape[0], dtype=np.float64)

    for x_idx in range(xs.shape[1]):
        for y_idx in range(ys.shape[1]):
            xx, yy = xs[:, [x_idx]], ys[:, y_idx]

            xx, yy = xx[~np.isnan(yy)], yy[~np.isnan(yy)]

            lm = LinearRegression().fit(xx, yy)

            yy_pred = lm.predict(xx)

            r2_scores[x_idx, y_idx] = r_squared(yy, yy_pred)

            log_likelihood(yy, yy_pred)

    return r2_scores


def r_squared(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred):
    sse = np.power(y_true - y_pred, 2).sum()
    sst = np.power(y_true - y_true.mean(), 2).sum()

    r = 1 - sse / sst

    return r


def log_likelihood(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred):
    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(-(np.power(y_true - y_pred, 2) / (2 * var)).sum())
    ln_l = np.log(l)

    return ln_l
