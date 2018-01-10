# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
cimport numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


"""
Linear regression omiting NaNs in dependent variable (ys)

"""
def lr(xs, ys, covs=None):
    outputs = __lr(xs.values, ys.values, covs)
    return {i: pd.DataFrame(outputs[i], index=xs.columns, columns=ys.columns) for i in outputs}


def __lr(xs, ys, covs):
    # Initialize variables
    cdef int x_idx, y_idx

    cdef np.ndarray xx = np.zeros(ys.shape[0], dtype=np.float64)
    cdef np.ndarray yy = np.zeros(ys.shape[0], dtype=np.float64)

    # Initialize output variables
    cdef np.ndarray betas = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    cdef np.ndarray r2_scores = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    cdef np.ndarray f_stat = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)
    cdef np.ndarray f_pval = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    # Iterate through matricies
    for x_idx in range(xs.shape[1]):

        for y_idx in range(ys.shape[1]):

            # Build arrays
            xx, yy = xs[:, [x_idx]], ys[:, y_idx]

            # Omit NaNs
            xx, yy = xx[~np.isnan(yy)], yy[~np.isnan(yy)]

            # Linear regression
            lm = LinearRegression().fit(xx, yy)

            # Predict
            yy_pred = lm.predict(xx)

            # Stats
            betas[x_idx, y_idx] = lm.coef_[0]

            r2_scores[x_idx, y_idx] = r_squared(yy, yy_pred)

            f_stats = f_statistic(yy, yy_pred, len(yy), xx.shape[1])
            f_stat[x_idx, y_idx] = f_stats[0]
            f_pval[x_idx, y_idx] = f_stats[1]

    # Assemble outputs
    res = {
        'beta': betas, 'r2': r2_scores, 'f_stat': f_stat, 'f_pval': f_pval
    }

    return res


"""
F-statistic
"""
def f_statistic(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred, np.int64_t n, np.int64_t p):
    msm = np.power(y_pred - y_true.mean(), 2).sum() / p
    mse = np.power(y_true - y_pred, 2).sum() / (n - p - 1)

    f = msm / mse

    f_pval = stats.f.sf(f, p, n - p - 1)

    return [f, f_pval]


"""
Coefficient of determination (r^2)

"""
def r_squared(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred):
    sse = np.power(y_true - y_pred, 2).sum()
    sst = np.power(y_true - y_true.mean(), 2).sum()

    r = 1 - sse / sst

    return r


"""
Log-likelihood

"""
def log_likelihood(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred):
    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(-(np.power(y_true - y_pred, 2) / (2 * var)).sum())
    ln_l = np.log(l)

    return ln_l
