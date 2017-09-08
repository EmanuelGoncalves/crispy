"""
This module implements all the necessary functions to perform genome-wide fitting of colocalized biases
of CRISPR dropout effects.

Copyright (C) 2017 Emanuel Goncalves
"""

# TODO: Finalise docs

import pandas as pd
from functools import partial
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from sklearn.gaussian_process import GaussianProcessRegressor
from limix_core.util.preprocess import covar_rescaling_factor_efficient
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic, PairwiseKernel


class DataCorrectionPosition(object):

    def __init__(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()

    def _correct(self, c, chrm, length_scale, alpha, n_restarts_optimizer, normalize_y, scale_pos):
        print('[%s] GP for variance decomposition started: CHRM %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c))

        # Get data-frames
        X_original, Y = self.X.loc[chrm[chrm == c].index].copy(), self.Y.loc[chrm[chrm == c].index].iloc[:, 0].copy()

        # Scale positions
        X = X_original / scale_pos

        # Instanciate the covariance functions
        K = ConstantKernel() * RationalQuadratic(length_scale_bounds=length_scale, alpha_bounds=alpha) + WhiteKernel()

        # Instanciate a Gaussian Process model
        gp = GaussianProcessRegressor(K, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, Y)
        print('[%s] CHRM: %s; GP: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c, gp.kernel_))

        # Predict with fitted kernel
        k_mean, k_se = gp.predict(X, return_std=True)

        # Calculate variance explained
        kernels = [('RQ', gp.kernel_.k1), ('Noise', gp.kernel_.k2)]

        cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(k.__call__(X))) for n, k in kernels}, name='Variance')
        cov_var /= cov_var.sum()

        # Build solution data-frame
        res = pd.concat([X_original, Y.rename('original')], axis=1)
        res = res.assign(mean=k_mean).assign(se=k_se).assign(chrm=c)
        res = res.assign(corrected=res['original'] - res['mean'])
        res = res.assign(var_rq=cov_var[kernels[0][0]]).assign(var_noise=cov_var[kernels[1][0]])

        return res

    def correct(self, chrm, chromosomes=None, length_scale=(1e-3, 10), alpha=(1e-3, 10), n_restarts_optimizer=3, normalize_y=True, scale_pos=1e9):
        chromosomes = set(chrm) if chromosomes is None else chromosomes

        with ThreadPool() as pool:
            res = pool.map(
                partial(
                    self._correct,
                    chrm=chrm,
                    length_scale=length_scale,
                    alpha=alpha,
                    n_restarts_optimizer=n_restarts_optimizer,
                    normalize_y=normalize_y,
                    scale_pos=scale_pos
                ), chromosomes
            )

        return pd.concat(res)


class DataCorrectionCopyNumber(object):

    def __init__(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()

    def _correct(self, c, chrm, metric, n_restarts_optimizer, normalize_y):
        print('[%s] GP for variance decomposition started: CHRM %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c))

        # Get data-frames
        X, Y = self.X.loc[chrm[chrm == c].index].copy(), self.Y.loc[chrm[chrm == c].index].iloc[:, 0].copy()

        # Instanciate the covariance functions
        K = ConstantKernel() * PairwiseKernel(metric=metric) + WhiteKernel()

        # Instanciate a Gaussian Process model
        gp = GaussianProcessRegressor(K, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, Y)
        print('[%s] CHRM: %s; GP: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c, gp.kernel_))

        # Predict with fitted kernel
        k_mean, k_se = gp.predict(X, return_std=True)

        # Calculate variance explained
        kernels = [(metric.upper(), gp.kernel_.k1), ('Noise', gp.kernel_.k2)]

        cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(k.__call__(X))) for n, k in kernels}, name='Variance')
        cov_var /= cov_var.sum()

        # Build solution data-frame
        res = pd.concat([X, Y.rename('original')], axis=1)
        res = res.assign(mean=k_mean).assign(se=k_se).assign(chrm=c)
        res = res.assign(corrected=res['original'] - res['mean'])
        res = res.assign(var_rq=cov_var[kernels[0][0]]).assign(var_noise=cov_var[kernels[1][0]])

        return res

    def correct(self, chrm, metric='rbf', chromosomes=None, n_restarts_optimizer=3, normalize_y=True):
        chromosomes = set(chrm) if chromosomes is None else chromosomes

        with ThreadPool() as pool:
            res = pool.map(
                partial(
                    self._correct,
                    chrm=chrm,
                    metric=metric,
                    n_restarts_optimizer=n_restarts_optimizer,
                    normalize_y=normalize_y,
                ), chromosomes
            )

        return pd.concat(res)
