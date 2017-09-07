"""
This module implements all the necessary functions to perform genome-wide fitting of colocalized biases
of CRISPR dropout effects.

Copyright (C) 2017 Emanuel Goncalves
"""

# TODO: Finalise docs

import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic


class DataCorrection(object):

    def __init__(self, X, Y):
        self.X = X.values if type(X) == pd.DataFrame else X
        self.Y = Y.values if type(X) == pd.DataFrame else Y

    def scale_positions(self, scale_pos=1e9):
        self.X /= scale_pos
        return self

    def _correct_position(self, c, chrm, length_scale, alpha, n_restarts_optimizer, normalize_y):
        print('[%s] GP for variance decomposition started: CHRM %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c))

        # Instanciate the covariance functions
        K = ConstantKernel() * RationalQuadratic(length_scale_bounds=length_scale, alpha_bounds=alpha) + WhiteKernel()

        # Instanciate a Gaussian Process model
        gp = GaussianProcessRegressor(K, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(self.X[chrm == c], self.Y[chrm == c, 0])
        print(gp.kernel_)

        # Normalise data
        k_mean, k_se = gp.predict(self.X[chrm == c], return_std=True)

        return pd.DataFrame(
            np.array([self.Y[chrm == c, 0], k_mean, k_se, self.Y[chrm == c, 0] - k_mean]).T,
            index=chrm[chrm == c].index, columns=['original', 'mean', 'se', 'corrected']
        )

    def correct_position(self, chrm, chromosomes=None, length_scale=(1e-3, 10), alpha=(1e-3, 10), n_restarts_optimizer=3, normalize_y=True):
        chromosomes = set(chrm) if chromosomes is None else chromosomes

        with ThreadPool() as pool:
            res = pool.map(
                partial(self._correct_position, chrm=chrm, length_scale=length_scale, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y),
                chromosomes
            )

        return pd.concat(res)
