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

    def __init__(self, X, Y, X_test=None, Y_test=None, chrm_test=None):
        self.X = X.copy()
        self.Y = Y.copy()

        self.X_test = X_test.copy() if X_test is not None else None
        self.Y_test = Y_test.copy() if Y_test is not None else None
        self.chrm_test = chrm_test.copy() if chrm_test is not None else None

    def _correct(self, c, chrm, length_scale, alpha, n_restarts_optimizer, normalize_y, scale_pos, return_hypers):
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

        # Build solution data-frame
        res = pd.concat([X_original, Y.rename('original')], axis=1)
        res = res.assign(mean=k_mean).assign(se=k_se).assign(chrm=c)
        res = res.assign(corrected=res['original'] - res['mean'])

        # Out-of-sample prediction
        if (self.X_test is not None) and (self.Y_test is not None) and (self.chrm_test is not None):
            X_test_original = self.X_test.loc[self.chrm_test[self.chrm_test == c].index].copy()
            Y_test = self.Y_test.loc[self.chrm_test[self.chrm_test == c].index].iloc[:, 0].copy()

            X_test = X_test_original / scale_pos

            k_pred_mean, k_pred_se = gp.predict(X_test, return_std=True)

            res_osp = pd.concat([X_test_original, Y_test.rename('original')], axis=1)
            res_osp = res_osp.assign(mean=k_pred_mean).assign(se=k_pred_se).assign(chrm=c)
            res_osp = res_osp.assign(corrected=res_osp['original'] - res_osp['mean'])

            res = res.assign(osp=False).append(res_osp.assign(osp=True))

        # Calculate variance explained
        kernels = [('RQ', gp.kernel_.k1), ('Noise', gp.kernel_.k2)]

        cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(k.__call__(X))) for n, k in kernels}, name='Variance')
        cov_var /= cov_var.sum()

        res = res.assign(var_rq=cov_var[kernels[0][0]]).assign(var_noise=cov_var[kernels[1][0]])

        # Return optimised hypers
        if return_hypers:
            hypers = {'%s_%s' % (n, p.name): k.get_params()[p.name] for n, k in kernels for p in k.hyperparameters}

            res = res.assign(alpha=hypers['RQ_alpha'])
            res = res.assign(length_scale=hypers['RQ_length_scale'])
            res = res.assign(noise=hypers['Noise_noise_level'])
            res = res.assign(ll=gp.log_marginal_likelihood_value_)

        return res

    def correct(self, chrm, chromosomes=None, length_scale=(1e-4, 1), alpha=(1e-7, 1), n_restarts_optimizer=3, normalize_y=True, scale_pos=1e3, return_hypers=False):
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
                    scale_pos=scale_pos,
                    return_hypers=return_hypers,
                ), chromosomes
            )

        return pd.concat(res)

    def correct_np(self, chrm, chromosomes=None, length_scale=(1e-3, 10), alpha=(1e-3, 10), n_restarts_optimizer=3, normalize_y=True, scale_pos=1e9):
        chromosomes = set(chrm) if chromosomes is None else chromosomes

        res = [
            self._correct(
                c,
                chrm=chrm,
                length_scale=length_scale,
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                normalize_y=normalize_y,
                scale_pos=scale_pos
            ) for c in chromosomes
        ]

        return pd.concat(res)
