# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
cimport numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def lr(xs, ys, ws=None):
    """
    Linear regression omiting NaNs in dependent variable (ys)

    """
    outputs = __lr(xs.values, ys.values, ws.values if ws is not None else None)
    return {i: pd.DataFrame(outputs[i], index=xs.columns, columns=ys.columns) for i in outputs}


def __lr(xs, ys, ws):
    # Initialize variables
    cdef float lr
    cdef int x_idx, y_idx

    cdef np.ndarray xx = np.zeros(ys.shape[0], dtype=np.float64)
    cdef np.ndarray yy = np.zeros(ys.shape[0], dtype=np.float64)

    # Initialize output variables
    cdef np.ndarray betas = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    cdef np.ndarray r2_scores = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)
    cdef np.ndarray r2_scores_lr = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    cdef np.ndarray f_stat = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)
    cdef np.ndarray f_pval = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)

    cdef np.ndarray lr_stat = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)
    cdef np.ndarray lr_pval = np.zeros([xs.shape[1], ys.shape[1]], dtype=np.float64)


    # Iterate through matricies
    for y_idx in range(ys.shape[1]):
        # Build Y array
        yy = ys[:, y_idx]

        # Remove NaNs
        yy_nans = np.isnan(yy)
        yy = yy[~yy_nans]

        # Linear regression only with covariates
        if ws is not None:
            covs = ws[~yy_nans]

            lm_pred_covs = LinearRegression().fit(covs, yy).predict(covs)

            lm_llog_covs = log_likelihood(yy, lm_pred_covs)

        for x_idx in range(xs.shape[1]):
            # Build X matrix
            xx = xs[:, [x_idx]][~yy_nans]

            # Linear regression
            lm = LinearRegression().fit(xx, yy)

            # Predict
            yy_pred = lm.predict(xx)

            # Betas
            betas[x_idx, y_idx] = lm.coef_[0]

            # R-squared
            r2_scores[x_idx, y_idx] = r_squared(yy, yy_pred)

            # F-statistic
            f_stats = f_statistic(yy, yy_pred, len(yy), xx.shape[1])
            f_stat[x_idx, y_idx] = f_stats[0]
            f_pval[x_idx, y_idx] = f_stats[1]

            if ws is not None:
                # Log-ratio test
                xx_covs = np.concatenate([xx, covs], axis=1)

                lm_pred_xx_covs = LinearRegression().fit(xx_covs, yy).predict(xx_covs)
                lm_llog_feat = log_likelihood(yy, lm_pred_xx_covs)

                lr = 2 * (lm_llog_feat - lm_llog_covs)
                lr_stat[x_idx, y_idx] = lr
                lr_pval[x_idx, y_idx] = stats.chi2(1).sf(lr)
                r2_scores_lr[x_idx, y_idx] = r_squared(yy, lm_pred_xx_covs)

    # Assemble outputs
    res = {
        'beta': betas, 'r2': r2_scores, 'f_stat': f_stat, 'f_pval': f_pval
    }

    if ws is not None:
        res.update({'lr': lr_stat, 'lr_pval': lr_pval, 'lr_r2': r2_scores_lr})

    return res


def f_statistic(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred, np.int64_t n, np.int64_t p):
    """
    F-statistic

    """

    msm = np.power(y_pred - y_true.mean(), 2).sum() / p
    mse = np.power(y_true - y_pred, 2).sum() / (n - p - 1)

    f = msm / mse

    f_pval = stats.f.sf(f, p, n - p - 1)

    return [f, f_pval]


def r_squared(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred):
    """
    Coefficient of determination (r^2)

    """

    sse = np.power(y_true - y_pred, 2).sum()
    sst = np.power(y_true - y_true.mean(), 2).sum()

    r = 1 - sse / sst

    return r


def log_likelihood(np.ndarray[np.float64_t, ndim=1] y_true, np.ndarray[np.float64_t, ndim=1] y_pred):
    """
    Log-likelihood

    """

    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(-(np.power(y_true - y_pred, 2) / (2 * var)).sum())
    ln_l = np.log(l)

    return ln_l
